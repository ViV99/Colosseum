from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable
from typing import Any

import numpy as np

from colosseum.envs.base_env import BaseEnv
from colosseum.envs.vec_env import VectorEnv

# ----------------------------------------------------------------------
# Worker-side commands (sent parent -> subprocess over the Pipe).
# ----------------------------------------------------------------------
_CMD_RESET_ALL = "reset_all"
_CMD_STEP = "step"
_CMD_RESET_DONE = "reset_done"
_CMD_CLOSE = "close"

# How long (seconds) to wait for a subprocess to join on close() before
# escalating to terminate().
_JOIN_TIMEOUT = 5.0


def _split_indices(num_envs: int, num_workers: int) -> list[tuple[int, int]]:
    """Partition ``range(num_envs)`` into ``num_workers`` contiguous slices.

    Returns a list of ``(start, end)`` half-open intervals.  The first
    ``num_envs % num_workers`` workers receive one extra env so the split is
    as balanced as possible.  Empty slices (when ``num_workers > num_envs``)
    are omitted, so the returned list may be shorter than ``num_workers``.
    """
    base, extra = divmod(num_envs, num_workers)
    slices: list[tuple[int, int]] = []
    start = 0
    for w in range(num_workers):
        size = base + (1 if w < extra else 0)
        if size == 0:
            continue
        slices.append((start, start + size))
        start += size
    return slices


def _worker_loop(
    conn,
    env_fn: Callable[[], BaseEnv],
    slice_size: int,
    global_offset: int,
) -> None:
    """Subprocess entry point: own a contiguous slice of envs via a ``VectorEnv``.

    This MUST be a top-level function so it is picklable under the ``spawn``
    start method.

    The worker manages ``slice_size`` envs (global indices
    ``global_offset .. global_offset + slice_size - 1``) using an in-process
    :class:`VectorEnv`, so all auto-reset / terminal_observation / stacking
    semantics are identical to the non-subprocess path.

    Args:
        conn: child end of a :func:`multiprocessing.Pipe`.
        env_fn: zero-arg factory producing one :class:`BaseEnv`.
        slice_size: number of envs this worker owns.
        global_offset: global index of this worker's first env (used to keep
            seeding globally consistent: local env ``j`` maps to global env
            ``global_offset + j``).
    """
    from colosseum.utils.logging import ENV_PROCESS_NAME, inherited_log_dir, setup_process_logging

    setup_process_logging(inherited_log_dir(), f"{os.environ.get(ENV_PROCESS_NAME, 'envproc')}-env{global_offset}")

    from colosseum.utils.process import init_child_process

    # Same policy as the worker: Ctrl-C is coordinated by the main process, and the
    # env process dies with its worker.
    init_child_process()

    import torch

    # Env children only step envs: one torch thread each (R2-04). OMP_NUM_THREADS=1
    # is already in this process's environment (set by the parent before spawn).
    torch.set_num_threads(1)
    vec_env = VectorEnv(env_fn, slice_size)
    try:
        while True:
            cmd, payload = conn.recv()

            if cmd == _CMD_STEP:
                result = vec_env.step(payload)
                conn.send(result)

            elif cmd == _CMD_RESET_ALL:
                seed = payload
                # Translate the global base seed into this slice's local base
                # seed.  VectorEnv adds the local index j, yielding a global
                # seed of (seed + global_offset) + j == seed + (global_offset + j).
                local_seed = (seed + global_offset) if seed is not None else None
                result = vec_env.reset_all(seed=local_seed)
                conn.send(result)

            elif cmd == _CMD_RESET_DONE:
                terminated, truncated = payload
                result = vec_env.reset_done(terminated, truncated)
                conn.send(result)

            elif cmd == _CMD_CLOSE:
                vec_env.close()
                conn.send(None)
                break

            else:  # pragma: no cover - defensive
                conn.send(RuntimeError(f"Unknown command: {cmd!r}"))
    except (EOFError, KeyboardInterrupt):
        # Parent went away or interrupted — shut down quietly.
        pass
    finally:
        try:
            vec_env.close()
        except Exception:
            pass
        conn.close()


class SubprocessVectorEnv:
    """Like :class:`VectorEnv`, but steps envs in parallel subprocesses.

    Drop-in replacement for :class:`colosseum.envs.vec_env.VectorEnv` exposing
    the identical public API (``num_envs``, ``num_players``,
    ``observation_space``, ``action_space``, ``action_spec`` plus
    ``reset_all`` / ``step`` / ``reset_done`` / ``close``) and returning the
    same stacked-numpy shapes and semantics.

    The ``num_envs`` envs are partitioned into ``num_workers`` contiguous
    slices, one slice per subprocess.  Each subprocess owns an in-process
    :class:`VectorEnv` over its slice, so auto-reset, ``terminal_observation``
    / ``terminal_info`` bookkeeping and observation stacking are byte-for-byte
    identical to the single-process implementation.  The parent only splits the
    action array by slice (axis 0) and concatenates per-worker results back
    along the env axis.

    Use this for CPU-heavy environments where sequential stepping in a single
    Python process is the throughput bottleneck.  For cheap envs the IPC /
    pickling overhead can outweigh the parallelism, so prefer the plain
    :class:`VectorEnv` there.
    """

    def __init__(
        self,
        env_fn: Callable[[], BaseEnv],
        num_envs: int,
        num_workers: int | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")

        if num_workers is None:
            num_workers = min(num_envs, os.cpu_count() or 1)
        num_workers = max(1, min(num_workers, num_envs))

        self.num_envs: int = num_envs

        # Read metadata (spaces / num_players) from one throwaway env in the
        # parent so the worker relies on exactly the same attributes as
        # VectorEnv.  Actual stepping happens only in subprocesses.
        probe_env = env_fn()
        self.num_players: int = probe_env.num_players
        self.observation_space = probe_env.observation_space
        self.action_space = probe_env.action_space
        probe_env.close()

        from colosseum.core.action_spec import ActionSpec
        self.action_spec: ActionSpec = ActionSpec.from_space(self.action_space)

        # Cache observation shape/dtype for pre-allocation (mirrors VectorEnv).
        _sample = self.observation_space.sample()
        self._obs_shape: tuple = np.asarray(_sample).shape
        self._obs_dtype = np.asarray(_sample).dtype

        # Contiguous env slices, one per worker.
        self._slices: list[tuple[int, int]] = _split_indices(num_envs, num_workers)
        self.num_workers: int = len(self._slices)

        # Spawn subprocesses.  spawn is required for macOS + torch and is what
        # the rest of the repo uses.
        self._ctx = mp.get_context("spawn")
        self._parent_conns: list[Any] = []
        self._procs: list[Any] = []
        self._closed = False

        # Children must start with OMP_NUM_THREADS=1: they import torch while
        # unpickling env_fn, before _worker_loop runs (R2-04).
        prev_omp = os.environ.get("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = "1"
        try:
            for (start, end) in self._slices:
                parent_conn, child_conn = self._ctx.Pipe()
                proc = self._ctx.Process(
                    target=_worker_loop,
                    args=(child_conn, env_fn, end - start, start),
                    daemon=True,
                )
                proc.start()
                # Close the child end in the parent so EOF propagates correctly.
                child_conn.close()
                self._parent_conns.append(parent_conn)
                self._procs.append(proc)
        finally:
            if prev_omp is None:
                os.environ.pop("OMP_NUM_THREADS", None)
            else:
                os.environ["OMP_NUM_THREADS"] = prev_omp

    # ------------------------------------------------------------------
    # Public API (mirrors VectorEnv exactly)
    # ------------------------------------------------------------------

    def reset_all(
        self, seed: int | None = None
    ) -> tuple[np.ndarray, list[dict[int, dict]]]:
        """Reset all envs.

        Args:
            seed: Optional base seed. Each env receives ``seed + i`` (over the
                global env index ``i``) if seed is not None, otherwise None —
                matching :meth:`VectorEnv.reset_all`.

        Returns:
            observations: np.ndarray of shape [num_envs, num_players, *obs_shape]
            infos: list of num_envs info dicts (each mapping player_index -> info)
        """
        for conn in self._parent_conns:
            conn.send((_CMD_RESET_ALL, seed))

        obs_parts: list[np.ndarray] = []
        all_infos: list[dict[int, dict]] = []
        for conn in self._parent_conns:
            obs_w, infos_w = self._recv(conn)
            obs_parts.append(obs_w)
            all_infos.extend(infos_w)

        observations = np.concatenate(obs_parts, axis=0)
        return observations, all_infos

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[int, dict]]]:
        """Step all envs and auto-reset any that are done.

        Args:
            actions: Array of shape [num_envs, num_players, *act_shape]
                (or [num_envs, num_players] for discrete action spaces).

        Returns:
            obs:        np.ndarray [num_envs, num_players, *obs_shape]
            rewards:    np.ndarray [num_envs, num_players]  (float64)
            terminated: np.ndarray [num_envs]  (bool) -- True if ANY player terminated
            truncated:  np.ndarray [num_envs]  (bool) -- True if ANY player truncated
            infos:      list of num_envs info dicts (player_index -> info).
                        For an env that auto-reset, each player's info is the
                        RESET info plus ``"terminal_observation"`` and
                        ``"terminal_info"``, identical to :meth:`VectorEnv.step`.
        """
        actions = np.asarray(actions)

        # Fan out: send each worker its contiguous slice of the action array.
        for conn, (start, end) in zip(self._parent_conns, self._slices):
            conn.send((_CMD_STEP, actions[start:end]))

        obs_parts: list[np.ndarray] = []
        rew_parts: list[np.ndarray] = []
        term_parts: list[np.ndarray] = []
        trunc_parts: list[np.ndarray] = []
        all_infos: list[dict[int, dict]] = []

        # Gather in slice order so concatenation matches global env indices.
        for conn in self._parent_conns:
            obs_w, rew_w, term_w, trunc_w, infos_w = self._recv(conn)
            obs_parts.append(obs_w)
            rew_parts.append(rew_w)
            term_parts.append(term_w)
            trunc_parts.append(trunc_w)
            all_infos.extend(infos_w)

        observations = np.concatenate(obs_parts, axis=0)
        rewards = np.concatenate(rew_parts, axis=0)
        terminated = np.concatenate(term_parts, axis=0)
        truncated = np.concatenate(trunc_parts, axis=0)
        return observations, rewards, terminated, truncated, all_infos

    def reset_done(
        self, terminated: np.ndarray, truncated: np.ndarray
    ) -> tuple[np.ndarray, list[dict[int, dict]]]:
        """Reset only the envs where the episode is done.

        Args:
            terminated: bool array [num_envs] -- per-env terminated flags.
            truncated:  bool array [num_envs] -- per-env truncated flags.

        Returns:
            obs:   np.ndarray [num_envs, num_players, *obs_shape].  For envs
                   that were NOT done, observations are zeros.
            infos: list of num_envs info dicts.  For envs that were NOT done,
                   the info dict is empty ``{}``.
        """
        terminated = np.asarray(terminated)
        truncated = np.asarray(truncated)

        for conn, (start, end) in zip(self._parent_conns, self._slices):
            conn.send((_CMD_RESET_DONE, (terminated[start:end], truncated[start:end])))

        obs_parts: list[np.ndarray] = []
        all_infos: list[dict[int, dict]] = []
        for conn in self._parent_conns:
            obs_w, infos_w = self._recv(conn)
            obs_parts.append(obs_w)
            all_infos.extend(infos_w)

        observations = np.concatenate(obs_parts, axis=0)
        return observations, all_infos

    def close(self) -> None:
        """Close all subprocesses and release resources.

        Sends a close sentinel to each worker, joins with a timeout, and
        escalates to ``terminate()`` for any straggler so this never hangs.
        Safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True

        # Politely ask each worker to close its envs and exit.
        for conn in self._parent_conns:
            try:
                conn.send((_CMD_CLOSE, None))
            except (BrokenPipeError, OSError, EOFError):
                pass

        # Drain the acknowledgement (best-effort) then close our pipe ends.
        for conn in self._parent_conns:
            try:
                if conn.poll(_JOIN_TIMEOUT):
                    conn.recv()
            except (EOFError, OSError):
                pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

        # Join, terminating any worker that did not exit in time.
        for proc in self._procs:
            proc.join(timeout=_JOIN_TIMEOUT)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=_JOIN_TIMEOUT)
            if proc.is_alive():  # pragma: no cover - last resort
                proc.kill()
                proc.join()

        self._parent_conns = []
        self._procs = []

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _recv(conn) -> Any:
        """Receive a worker reply, re-raising any exception it forwarded."""
        msg = conn.recv()
        if isinstance(msg, BaseException):
            raise msg
        return msg
