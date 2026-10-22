"""Recovery points of the learners (SP5 spec block 5.2; task T4.2).

``runs/<name>/recovery/<agent>/recovery.frame`` holds the newest recovery point of each agent: one checkpoint frame
(``colosseum.net.chunks.encode_checkpoint(..., kind="recovery")``: the model weights and the trainer state, the
latter as one uint8 leaf of ``torch.save`` bytes). The hub never decodes the arrays and never unpickles anything:
it checks the frame header only (``colosseum.net.frame.peek_meta``) and moves the spooled upload into place with
one ``os.replace`` (atomic), so a reader (a ``Download`` servicer thread, a resumed hub) sees the old or the new
point, never a mix. ``meta.json`` next to it is a readable copy (``{"generation", "version", "env_steps"}``) written
after the frame; the frame header is the authority when the two disagree (a crash between the two writes).

The directory is the spec's layout (``recovery/<agent>/``, next to ``checkpoints/<agent>/``), not a blob of
``runs/<name>/blobs/``: agent ids are the config's (``check_agent_id``: no path separators), and ``Download(kind=
"start", ident="recovery")`` serves ``RecoveryStore.path(agent)`` (``hub.server.BlobStore.start_path``).

A recovery point is not a snapshot: it is never in the opponent pool, never rated, never retained or evicted; it is
overwritten by the next one. Accepted only from the agent's current learner generation (spec 7.2) and never older
than the stored one. Main loop only (servicer threads only read ``path(agent)``).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from colosseum.net.frame import MAX_HEADER_BYTES, FrameError, peek_meta
from colosseum.utils.fs import write_text_atomic

logger = logging.getLogger(__name__)

__all__ = ["RECOVERY_DIR", "RECOVERY_FRAME", "RECOVERY_META", "RecoveryPoint", "RecoveryStore", "read_frame_meta"]

RECOVERY_DIR = "recovery"
RECOVERY_FRAME = "recovery.frame"
RECOVERY_META = "meta.json"
_HEAD = 16                         # the frame's fixed prefix: magic, version, flags, header length


@dataclass(frozen=True)
class RecoveryPoint:
    agent_id: str
    generation: int
    version: int
    env_steps: int
    size: int
    path: Path


def read_frame_meta(path: str | Path) -> tuple[str, dict]:
    """``peek_meta`` of a frame file (reads the header only). FrameError for a file that is not a frame."""
    with open(path, "rb") as fh:
        head = fh.read(_HEAD)
        if len(head) < _HEAD:
            raise FrameError("truncated")
        length = int.from_bytes(head[8:16], "little")
        if length > MAX_HEADER_BYTES:
            raise FrameError("header_too_large")
        return peek_meta(head + fh.read(length))


class RecoveryStore:
    """The recovery points of a run dir (module docstring)."""

    def __init__(self, run_root: str | Path) -> None:
        self._root = Path(run_root) / RECOVERY_DIR
        self._warned: set[Path] = set()      # unreadable points already logged (one WARNING per path)

    def path(self, agent_id: str) -> Path:
        return self._root / agent_id / RECOVERY_FRAME

    def get(self, agent_id: str) -> RecoveryPoint | None:
        """The stored point (from the frame header; ``env_steps`` from meta.json, 0 when missing or not of this
        point), or None (none yet, or an unreadable file: one WARNING per path)."""
        path = self.path(agent_id)
        try:
            if not path.is_file():
                return None
            kind, meta = read_frame_meta(path)
            if kind != "checkpoint" or meta.get("kind") != "recovery" or meta.get("agent_id") != agent_id:
                raise FrameError("not_a_recovery_point")
            generation, version = int(meta["generation"]), int(meta["policy_version"])
            size = path.stat().st_size
        except (OSError, FrameError, KeyError, TypeError, ValueError) as e:
            if path not in self._warned:
                self._warned.add(path)
                logger.warning(f"Recovery point {path} is unreadable ({type(e).__name__}: {e}); ignored")
            return None
        env_steps = 0
        try:
            saved = json.loads((path.parent / RECOVERY_META).read_text())
            if int(saved.get("version", -1)) == version and int(saved.get("generation", -1)) == generation:
                env_steps = int(saved.get("env_steps") or 0)
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        return RecoveryPoint(agent_id, generation, version, env_steps, size, path)

    def accept(self, agent_id: str, spooled: str | Path, *, generation: int, version: int, env_steps: int,
               current_generation: int) -> tuple[bool, str]:
        """Store the uploaded frame ``spooled`` (in the run dir's ``.incoming/``) as ``agent_id``'s recovery point.
        Returns ``(accepted, reason)``; reasons: ``stale_generation`` (not the agent's current learner
        generation), ``invalid`` (the header is not a recovery checkpoint of this agent, generation and version),
        ``not_newer`` (older than the stored point). A refused file is deleted."""
        spooled = Path(spooled)
        reason = self._check(agent_id, spooled, generation, version, current_generation)
        if reason:
            spooled.unlink(missing_ok=True)
            return False, reason
        target = self.path(agent_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(spooled, target)
        self._warned.discard(target)
        write_text_atomic(target.parent / RECOVERY_META, json.dumps(
            {"generation": int(generation), "version": int(version), "env_steps": int(env_steps)}) + "\n")
        return True, ""

    def _check(self, agent_id: str, spooled: Path, generation: int, version: int, current_generation: int) -> str:
        if int(generation) != int(current_generation):
            return "stale_generation"
        try:
            kind, meta = read_frame_meta(spooled)
        except (OSError, FrameError):
            return "invalid"
        if (kind != "checkpoint" or meta.get("kind") != "recovery" or meta.get("agent_id") != agent_id
                or meta.get("generation") != generation or meta.get("policy_version") != version):
            return "invalid"
        stored = self.get(agent_id)
        if stored is not None and version < stored.version:
            return "not_newer"
        return ""
