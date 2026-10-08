"""In-process rollout loop: a MatchRunner plus agent-owned chunk v2 buffers (spec blocks 4-5).

All I/O goes through :class:`LoopIO` callbacks, so the loop runs unchanged in a worker
process (``rollout_worker_process``) and in-process in tests.

``RolloutLoop`` is the :class:`MatchRunner`'s model pool (each agent's latest model plus
frozen checkpoints) and its observer. As observer it writes the slots of every collecting
seat (slot rules of spec block 4):

- **The seat acts.** With an open ACT and exactly one free slot: a BOOT with the current
  observation and ``global_state``, the chunk is sealed, and the new ACT goes into the same
  buffer after the reset (never into a parked buffer). Without an open ACT and exactly one
  free slot: a PAD, seal, then the ACT. Otherwise the ACT goes into the next slot (the
  previous transition closes implicitly). The first ACT carries the seat's pending reward.
- **Rewards** go to the seat's open ACT, or to its pending reward before its first ACT of
  the episode.
- **Elimination, or an episode that ends by the rules:** the open ACT becomes terminal.
- **Truncation:** a BOOT with ``final_obs`` / final ``global_state`` and ``reset_after``
  follows the open ACT.
- A chunk is sealed as soon as its last slot is taken (always a BOOT or a PAD).
- A seat that never acted in an episode drops its pending reward at the episode end
  (counted in ``stats["dropped_reward_episodes"]``).
- Buffers are owned by agents and parked when a lineup change at an episode end stops a
  seat from collecting for its agent; a seat that starts collecting takes a parked buffer
  of that agent first (``BufferPool``).
- A chunk's ``policy_version`` is the agent's version at its first slot.
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from colosseum.sp2.core.roles import agent_role_spec
from colosseum.sp2.core.specs import ActionSpec, ObsSpec
from colosseum.sp2.core.types import (
    LATEST_NETWORK_ID,
    Lineup,
    MatchResult,
    TrajectoryChunk,
    WeightPayload,
    WorkerCommand,
    state_dict_from_numpy,
)
from colosseum.sp2.envs.game import MultiAgentEnv
from colosseum.sp2.envs.vector import SubprocessVectorEnv, VectorEnv
from colosseum.sp2.networks.model import PolicyModel
from colosseum.sp2.worker.buffers import BufferPool, BufferSpec, RolloutBuffer
from colosseum.sp2.worker.match_runner import ActRecord, EpisodeEnd, MatchRunner

logger = logging.getLogger(__name__)

__all__ = ["LATEST_NETWORK_ID", "LoopIO", "RolloutLoop"]


@dataclass
class LoopIO:
    """All I/O of the rollout loop goes through these callbacks."""

    send_chunk: Callable[[TrajectoryChunk], None]
    poll_weights: Callable[[str], WeightPayload | None]
    report_result: Callable[[MatchResult], None] | None = None
    poll_command: Callable[[], WorkerCommand | None] | None = None
    add_env_steps: Callable[[int], None] | None = None  # global env-step budget counter


@dataclass
class _SeatTrack:
    """Rollout state of one collecting (env, seat)."""

    agent_id: str
    buffer: RolloutBuffer
    pending_reward: float = 0.0


class RolloutLoop:
    """One worker's rollout loop over ``num_envs`` envs (module docstring).

    ``lineups`` holds exactly one lineup per env (``ValueError`` before any env is built);
    a ``WorkerCommand`` may carry fewer lineups than envs (missing = unchanged), never more.
    """

    def __init__(
        self,
        *,
        worker_id: int,
        env_fn: Callable[[], MultiAgentEnv],
        num_envs: int,
        chunk_length: int,
        agent_ids: list[str],
        agent_roles: Mapping[str, Sequence[str]],
        model_factories: Mapping[str, Callable[[], PolicyModel]],
        io: LoopIO,
        lineups: Sequence[Lineup],
        weight_sync_interval: float = 5.0,
        checkpoint_state_dicts_by_agent: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]] | None = None,
        seed: int | None = None,
        vec_env_kind: Literal["sync", "subprocess"] = "sync",
        subproc_workers: int | None = None,
        max_idle_steps: int = 1000,
    ) -> None:
        if len(lineups) != num_envs:
            raise ValueError(f"worker {worker_id}: need one lineup per env: {len(lineups)} lineups for {num_envs} envs")
        self.worker_id = worker_id
        self._io = io
        self._agent_ids = list(agent_ids)
        self._model_factories = dict(model_factories)
        self._weight_sync_interval = weight_sync_interval

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        if vec_env_kind == "subprocess":
            vec_env = SubprocessVectorEnv(env_fn, num_envs, num_workers=subproc_workers)
        elif vec_env_kind == "sync":
            vec_env = VectorEnv(env_fn, num_envs)
        else:
            raise ValueError(f"unknown vec_env_kind {vec_env_kind!r}")
        try:
            spec = vec_env.spec

            specs: dict[str, BufferSpec] = {}
            for aid in self._agent_ids:
                role = agent_role_spec(spec, list(agent_roles[aid]))
                specs[aid] = BufferSpec(
                    obs=ObsSpec.from_space(role.observation_space),
                    action=ActionSpec.from_space(role.action_space),
                    global_state=(
                        None if role.global_state_space is None else ObsSpec.from_space(role.global_state_space)
                    ),
                )
            self._pool = BufferPool(chunk_length, specs)

            # Model pool: agent -> network id -> model ("latest" + frozen checkpoints).
            self._models: dict[str, dict[str, PolicyModel]] = {}
            self._policy_versions: dict[str, int] = {aid: 0 for aid in self._agent_ids}
            for aid in self._agent_ids:
                latest = self._model_factories[aid]()
                latest.eval()
                self._models[aid] = {LATEST_NETWORK_ID: latest}
            for aid, by_id in (checkpoint_state_dicts_by_agent or {}).items():
                for ckpt_id, sd in by_id.items():
                    self._add_checkpoint(aid, ckpt_id, sd)
            self.sync_weights()   # initial weights and policy versions

            self._tracks: list[dict[int, _SeatTrack]] = [{} for _ in range(num_envs)]
            self._env_steps = 0
            self._episodes = 0
            self._chunks_sent = 0
            self._dropped_reward_episodes = 0
            self._recorded: dict[str, int] = defaultdict(int)
            self._runner = MatchRunner(
                vec_env=vec_env, lineups=lineups, models=self, observer=self, seed=seed,
                max_idle_steps=max_idle_steps, context=f"worker {worker_id}, ", match_id_prefix=f"w{worker_id}_e",
            )
            for e in range(num_envs):
                for seat, assignment in enumerate(self._runner.lineup(e).seats):
                    if assignment.collect:
                        self._start_collecting(e, seat, assignment.agent_id)
        except BaseException:
            # the caller never gets this loop, so nobody else would close the envs (subprocesses!)
            try:
                vec_env.close()
            except Exception:
                logger.warning(f"Worker {worker_id}: closing the vector env after a failed setup failed",
                               exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> int:
        """One step of every env. Returns the env steps taken."""
        self._poll_command()
        n = self._runner.step()
        self._env_steps += n
        if self._io.add_env_steps is not None:
            self._io.add_env_steps(n)
        if time.monotonic() - self._last_weight_sync >= self._weight_sync_interval:
            self.sync_weights()
        return n

    def sync_weights(self) -> None:
        """Pull the newest weights for every agent's latest model (non-blocking)."""
        for aid in self._agent_ids:
            payload = self._io.poll_weights(aid)
            if payload is not None:
                self._models[aid][LATEST_NETWORK_ID].load_state_dict(payload.to_torch_state_dict())
                self._policy_versions[aid] = int(payload.policy_version)
        self._last_weight_sync = time.monotonic()

    def run(self, should_stop: Callable[[], bool], max_env_steps: int = 0) -> None:
        """Step until ``should_stop()`` or until ``max_env_steps`` env steps (0 = no limit)."""
        while not should_stop():
            if max_env_steps > 0 and self._env_steps >= max_env_steps:
                break
            self.step()

    def close(self) -> None:
        self._runner.close()

    @property
    def stats(self) -> dict[str, int]:
        out = {
            "chunks_sent": self._chunks_sent,
            "env_steps": self._env_steps,
            "episodes": self._episodes,
            "parked_buffers": self._pool.parked_count(),
            "dropped_reward_episodes": self._dropped_reward_episodes,
            "recorded_transitions": sum(self._recorded.values()),
        }
        for aid in self._agent_ids:
            out[f"recorded_transitions/{aid}"] = self._recorded[aid]
            out[f"buffered_transitions/{aid}"] = self._buffered_acts(aid)
        return out

    def buffered_reward(self, agent_id: str) -> float:
        """Rewards of ``agent_id`` recorded in buffers but not sent yet (seat + parked buffers)."""
        total = self._pool.parked_reward(agent_id)
        for tracks in self._tracks:
            for track in tracks.values():
                if track.agent_id == agent_id:
                    total += track.buffer.reward_sum
        return total

    # ------------------------------------------------------------------
    # ModelPool
    # ------------------------------------------------------------------

    def get(self, agent_id: str, network_id: str) -> PolicyModel | None:
        return self._models.get(agent_id, {}).get(network_id)

    def _add_checkpoint(self, aid: str, ckpt_id: str, sd: Mapping[str, np.ndarray]) -> None:
        if aid not in self._models or ckpt_id in self._models[aid]:
            return
        model = self._model_factories[aid]()
        model.load_state_dict(state_dict_from_numpy(sd))
        model.eval()
        self._models[aid][ckpt_id] = model
        logger.info(f"Worker {self.worker_id}: agent {aid}: loaded checkpoint {ckpt_id}")

    def _poll_command(self) -> None:
        """Load a command's new checkpoints now; stage its lineups for the envs' episode ends."""
        if self._io.poll_command is None:
            return
        cmd = self._io.poll_command()
        if cmd is None:
            return
        for aid, ckpts in cmd.new_checkpoints.items():
            for ckpt_id, sd in ckpts.items():
                self._add_checkpoint(aid, ckpt_id, sd)
        if len(cmd.lineups) > self._runner.num_envs:
            raise ValueError(
                f"worker {self.worker_id}: a command has {len(cmd.lineups)} lineups for "
                f"{self._runner.num_envs} envs"
            )
        for e, lineup in enumerate(cmd.lineups):
            if lineup is not None:
                self._runner.set_next_lineup(e, lineup)

    # ------------------------------------------------------------------
    # MatchObserver
    # ------------------------------------------------------------------

    def on_act(self, env: int, seat: int, record: ActRecord) -> None:
        track = self._tracks[env].get(seat)
        if track is None:
            return
        buf = track.buffer
        if buf.free_slots == 1:
            if buf.has_open:
                buf.write_boot(record.obs, record.global_state, reset_after=False)
            else:
                buf.write_pad()
            self._seal(track.agent_id, buf)
        if buf.slots_used == 0:
            buf.begin(record.pre_state, self._policy_versions[track.agent_id])
        buf.write_act(record.obs, record.global_state, record.mask, record.action, record.log_prob,
                      record.unit_log_probs, track.pending_reward)
        track.pending_reward = 0.0
        self._recorded[track.agent_id] += 1

    def on_rewards(self, env: int, rewards: dict[int, float]) -> None:
        for seat, r in rewards.items():
            track = self._tracks[env].get(seat)
            if track is None:
                continue
            if track.buffer.has_open:
                track.buffer.add_reward(r)
            else:
                track.pending_reward += r

    def on_terminated(self, env: int, seats: list[int]) -> None:
        for seat in seats:
            track = self._tracks[env].get(seat)
            if track is not None and track.buffer.has_open:
                track.buffer.mark_terminal()

    def on_episode_end(self, env: int, end: EpisodeEnd) -> None:
        for seat, track in self._tracks[env].items():
            buf = track.buffer
            if buf.has_open:
                if end.truncated:
                    gs = None if end.final_global_state is None else end.final_global_state.get(seat)
                    buf.write_boot(end.final_obs[seat], gs, reset_after=True)
                    if buf.is_full:
                        self._seal(track.agent_id, buf)
                else:
                    buf.mark_terminal()
            if track.pending_reward != 0.0:
                self._dropped_reward_episodes += 1
                logger.debug(
                    f"Worker {self.worker_id}: env {env}, seat {seat}: dropping reward "
                    f"{track.pending_reward} of an episode the seat never acted in"
                )
            track.pending_reward = 0.0
        self._episodes += 1
        if self._io.report_result is not None:
            self._io.report_result(end.result)

    def on_lineup_applied(self, env: int, old: Lineup, new: Lineup) -> None:
        tracks = self._tracks[env]
        for seat in range(max(len(old.seats), len(new.seats))):
            assignment = new.seats[seat] if seat < len(new.seats) else None
            collector = assignment.agent_id if assignment is not None and assignment.collect else None
            track = tracks.get(seat)
            if track is not None and track.agent_id != collector:
                self._pool.park(track.agent_id, track.buffer)
                del tracks[seat]
            if collector is not None and seat not in tracks:
                self._start_collecting(env, seat, collector)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _start_collecting(self, env: int, seat: int, agent_id: str) -> None:
        if agent_id not in self._models:
            raise ValueError(f"worker {self.worker_id}: env {env}, seat {seat}: unknown agent {agent_id!r}")
        self._tracks[env][seat] = _SeatTrack(agent_id=agent_id, buffer=self._pool.acquire(agent_id))

    def _buffered_acts(self, aid: str) -> int:
        n = self._pool.parked_acts(aid)
        for tracks in self._tracks:
            for track in tracks.values():
                if track.agent_id == aid:
                    n += track.buffer.num_acts
        return n

    def _seal(self, aid: str, buf: RolloutBuffer) -> None:
        chunk = buf.build_chunk(aid)
        buf.reset()
        self._io.send_chunk(chunk)
        self._chunks_sent += 1
