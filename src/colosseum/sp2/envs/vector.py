"""Vectors of ``MultiAgentEnv``s without auto-reset (SP2 spec block 1, "Вектор-среды").

The caller (``MatchRunner``) decides when and how each env is reset: ``step`` only steps
the envs it is given actions for and returns their results as they are (a finished env
returns its final result); ``reset`` resets the requested envs with ``(seed, layout)``.

``SubprocessVectorEnv`` runs contiguous slices of envs in child processes. Each
``reset()`` / ``step()`` call is one IPC round with the children that own a requested env,
so all resets of a step are batched into one round. Exceptions raised in a child (e.g. by
the env) are re-raised in the parent with the child's traceback as a note.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from collections.abc import Callable, Mapping
from typing import Any

from colosseum.core.errors import EnvContractError
from colosseum.sp2.envs.game import GameSpec, MultiAgentEnv, StepResult

_CMD_SPEC = "spec"
_CMD_RESET = "reset"
_CMD_STEP = "step"
_CMD_CLOSE = "close"
_JOIN_TIMEOUT = 5.0


def _env_spec(env: Any, index: int) -> GameSpec:
    if not isinstance(env, MultiAgentEnv):
        raise EnvContractError(f"env {index}: env_fn must return a MultiAgentEnv, got {type(env).__name__}")
    spec = getattr(env, "spec", None)
    if not isinstance(spec, GameSpec):
        raise EnvContractError(f"env {index}: {type(env).__name__}.spec must be a GameSpec, "
                               f"got {type(spec).__name__}")
    return spec


class VectorEnv:
    """``num_envs`` envs in this process, stepped one after another."""

    def __init__(self, env_fn: Callable[[], MultiAgentEnv], num_envs: int, *, index_offset: int = 0) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.num_envs = num_envs
        self._offset = index_offset
        self.envs: list[MultiAgentEnv] = []
        try:
            for i in range(num_envs):
                env = env_fn()
                self.envs.append(env)
                spec = _env_spec(env, index_offset + i)
                if i == 0:
                    spec.validate()
                    self.spec: GameSpec = spec
                elif spec != self.spec:
                    raise EnvContractError(f"env {index_offset + i} reports a different GameSpec than env "
                                           f"{index_offset}; every env of a vector must have the same spec")
        except BaseException:
            self.close()
            raise

    def _env(self, index: int) -> MultiAgentEnv:
        if not 0 <= index < self.num_envs:
            raise IndexError(f"env index {index} out of range 0..{self.num_envs - 1}")
        return self.envs[index]

    def reset(self, requests: Mapping[int, tuple[int | None, str]]) -> dict[int, StepResult]:
        """Reset the listed envs: ``{env: (seed, layout)}`` -> ``{env: reset result}``."""
        out: dict[int, StepResult] = {}
        for index, (seed, layout) in requests.items():
            if layout not in self.spec.layouts:
                raise ValueError(f"env {self._offset + index}: unknown layout {layout!r} "
                                 f"(layouts: {list(self.spec.layouts)})")
            out[index] = self._env(index).reset(seed, layout)
        return out

    def step(self, actions: Mapping[int, dict[int, Any]]) -> dict[int, StepResult]:
        """Step the listed envs only: ``{env: {seat: action}}`` -> ``{env: result}``."""
        return {index: self._env(index).step(dict(seat_actions)) for index, seat_actions in actions.items()}

    def close(self) -> None:
        envs, self.envs = self.envs, []
        for env in envs:
            close = getattr(env, "close", None)
            if callable(close):
                close()


def _split_indices(num_envs: int, num_workers: int) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` slices; the first ``num_envs % num_workers`` get one more env."""
    base, extra = divmod(num_envs, num_workers)
    slices, start = [], 0
    for w in range(num_workers):
        size = base + (1 if w < extra else 0)
        if size:
            slices.append((start, start + size))
            start += size
    return slices


def _with_child_traceback(exc: BaseException) -> BaseException:
    exc.add_note("raised in a SubprocessVectorEnv child:\n" + "".join(traceback.format_exception(exc)))
    return exc


def _worker_loop(conn: Any, env_fn: Callable[[], MultiAgentEnv], num_envs: int, offset: int) -> None:
    """Child entry point (top level, so spawn can pickle it): owns envs ``offset .. offset+num_envs-1``."""
    from colosseum.utils.logging import ENV_PROCESS_NAME, inherited_log_dir, setup_process_logging

    setup_process_logging(inherited_log_dir(), f"{os.environ.get(ENV_PROCESS_NAME, 'envproc')}-env{offset}")

    from colosseum.utils.process import init_child_process

    init_child_process()  # Ctrl-C is coordinated by the main process; die with the parent

    import torch

    torch.set_num_threads(1)  # env children only step envs; OMP_NUM_THREADS=1 comes from the parent
    vec: VectorEnv | None = None
    startup_error: BaseException | None = None
    try:
        vec = VectorEnv(env_fn, num_envs, index_offset=offset)
    except Exception as e:  # noqa: BLE001 - reported on the first command
        startup_error = _with_child_traceback(e)
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == _CMD_CLOSE:
                conn.send(None)
                break
            if startup_error is not None:
                conn.send(startup_error)
                continue
            try:
                if cmd == _CMD_SPEC:
                    reply: Any = vec.spec
                elif cmd == _CMD_RESET:
                    reply = vec.reset(payload)
                elif cmd == _CMD_STEP:
                    reply = vec.step(payload)
                else:
                    reply = RuntimeError(f"unknown SubprocessVectorEnv command {cmd!r}")
            except Exception as e:  # noqa: BLE001 - forwarded to the parent
                reply = _with_child_traceback(e)
            try:
                conn.send(reply)
            except Exception as e:  # noqa: BLE001 - e.g. an unpicklable exception or result
                conn.send(RuntimeError(f"cannot send the reply of {cmd!r} to the parent: {type(e).__name__}: {e}; "
                                       f"original reply: {reply!r}"))
    except (EOFError, KeyboardInterrupt):
        pass  # the parent went away
    finally:
        if vec is not None:
            try:
                vec.close()
            except Exception:  # noqa: BLE001 - best effort
                pass
        conn.close()


class SubprocessVectorEnv:
    """Same interface as :class:`VectorEnv`; envs live in ``num_workers`` child processes."""

    def __init__(self, env_fn: Callable[[], MultiAgentEnv], num_envs: int, num_workers: int | None = None) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if num_workers is None:
            num_workers = min(num_envs, os.cpu_count() or 1)
        num_workers = max(1, min(num_workers, num_envs))
        self.num_envs = num_envs
        probe = env_fn()  # the spec is read in the parent; children compare theirs with it
        try:
            self.spec: GameSpec = _env_spec(probe, 0)
            self.spec.validate()
        finally:
            probe.close()
        self._slices = _split_indices(num_envs, num_workers)
        self.num_workers = len(self._slices)
        self._owner = [w for w, (start, end) in enumerate(self._slices) for _ in range(start, end)]
        self._ctx = mp.get_context("spawn")
        self._conns: list[Any] = []
        self._procs: list[Any] = []
        self._closed = False
        prev_omp = os.environ.get("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = "1"  # children import torch while unpickling env_fn
        try:
            for start, end in self._slices:
                parent_conn, child_conn = self._ctx.Pipe()
                proc = self._ctx.Process(target=_worker_loop, args=(child_conn, env_fn, end - start, start),
                                         daemon=True)
                proc.start()
                child_conn.close()
                self._conns.append(parent_conn)
                self._procs.append(proc)
        finally:
            if prev_omp is None:
                os.environ.pop("OMP_NUM_THREADS", None)
            else:
                os.environ["OMP_NUM_THREADS"] = prev_omp
        try:
            for w, spec in self._round({w: (_CMD_SPEC, None) for w in range(self.num_workers)}).items():
                if spec != self.spec:
                    start = self._slices[w][0]
                    raise EnvContractError(f"env {start} (child {w}) reports a different GameSpec than the env "
                                           f"built in the parent; every env must have the same spec")
        except BaseException:
            self.close()
            raise

    def _round(self, messages: Mapping[int, tuple[str, Any]]) -> dict[int, Any]:
        """Send one message per listed child, then collect every reply (re-raising child errors)."""
        for w, message in messages.items():
            self._conns[w].send(message)
        replies = {w: self._conns[w].recv() for w in messages}
        for reply in replies.values():
            if isinstance(reply, BaseException):
                raise reply
        return replies

    def _by_child(self, per_env: Mapping[int, Any]) -> dict[int, dict[int, Any]]:
        out: dict[int, dict[int, Any]] = {}
        for index, value in per_env.items():
            if not 0 <= index < self.num_envs:
                raise IndexError(f"env index {index} out of range 0..{self.num_envs - 1}")
            w = self._owner[index]
            out.setdefault(w, {})[index - self._slices[w][0]] = value
        return out

    def _merge(self, replies: Mapping[int, dict[int, StepResult]]) -> dict[int, StepResult]:
        return {self._slices[w][0] + local: result for w, part in replies.items() for local, result in part.items()}

    def reset(self, requests: Mapping[int, tuple[int | None, str]]) -> dict[int, StepResult]:
        for index, (_seed, layout) in requests.items():
            if layout not in self.spec.layouts:
                raise ValueError(f"env {index}: unknown layout {layout!r} (layouts: {list(self.spec.layouts)})")
        parts = self._by_child(requests)
        return self._merge(self._round({w: (_CMD_RESET, part) for w, part in parts.items()}))

    def step(self, actions: Mapping[int, dict[int, Any]]) -> dict[int, StepResult]:
        parts = self._by_child(actions)
        return self._merge(self._round({w: (_CMD_STEP, part) for w, part in parts.items()}))

    def close(self) -> None:
        """Close every child (join with a timeout, then terminate/kill); safe to call twice."""
        if self._closed:
            return
        self._closed = True
        for conn in self._conns:
            try:
                conn.send((_CMD_CLOSE, None))
            except (BrokenPipeError, OSError, EOFError):
                pass
        for conn in self._conns:
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
        for proc in self._procs:
            proc.join(timeout=_JOIN_TIMEOUT)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=_JOIN_TIMEOUT)
            if proc.is_alive():  # pragma: no cover - last resort
                proc.kill()
                proc.join()
        self._conns, self._procs = [], []

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
