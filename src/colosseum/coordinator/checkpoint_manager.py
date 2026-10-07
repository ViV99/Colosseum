"""Checkpoint manager: save, load, and FIFO-manage model checkpoints."""

from __future__ import annotations

import json
import logging
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """Metadata for a saved checkpoint."""

    checkpoint_id: str
    agent_id: str
    policy_version: int
    path: str
    timestamp: float
    metrics: dict = field(default_factory=dict)


class CheckpointManager:
    """Saves checkpoints, manages FIFO pool per agent.

    Directory structure:
        base_dir/
            agent_0/
                ckpt_v100/
                    model.pt
                    meta.json
                ckpt_v200/
                    ...
    """

    def __init__(
        self,
        base_dir: str,
        pool_size: int = 20,
        save_optimizer: bool = True,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._pool_size = pool_size
        self._save_optimizer = save_optimizer
        self._base_dir.mkdir(parents=True, exist_ok=True)

        # In-memory index: agent_id -> list of CheckpointInfo (sorted by version)
        self._index: dict[str, list[CheckpointInfo]] = {}
        self._scan_existing()

    def _scan_existing(self) -> None:
        """Scan disk for existing checkpoints on startup."""
        if not self._base_dir.exists():
            return
        for agent_dir in self._base_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            agent_id = agent_dir.name
            self._index[agent_id] = []
            for ckpt_dir in sorted(agent_dir.iterdir()):
                meta_path = ckpt_dir / "meta.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    info = CheckpointInfo(**meta)
                    self._index[agent_id].append(info)
            self._index[agent_id].sort(key=lambda c: c.policy_version)

    def save(
        self,
        agent_id: str,
        policy_version: int,
        state_dict: dict,
        optimizer_state: dict | None = None,
        metrics: dict | None = None,
    ) -> str:
        """Save a checkpoint. Returns checkpoint_id. Evicts oldest if pool full."""
        checkpoint_id = f"ckpt_v{policy_version}"
        agent_dir = self._base_dir / agent_id
        ckpt_dir = agent_dir / checkpoint_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(state_dict, ckpt_dir / "model.pt")

        # Save optimizer if requested
        if self._save_optimizer and optimizer_state is not None:
            torch.save(optimizer_state, ckpt_dir / "optimizer.pt")

        # Save metadata
        info = CheckpointInfo(
            checkpoint_id=checkpoint_id,
            agent_id=agent_id,
            policy_version=policy_version,
            path=str(ckpt_dir),
            timestamp=time.time(),
            metrics=metrics or {},
        )
        with open(ckpt_dir / "meta.json", "w") as f:
            json.dump(asdict(info), f, indent=2)

        # Update index
        if agent_id not in self._index:
            self._index[agent_id] = []
        self._index[agent_id].append(info)

        # FIFO eviction
        while len(self._index[agent_id]) > self._pool_size:
            oldest = self._index[agent_id].pop(0)
            oldest_path = Path(oldest.path)
            if oldest_path.exists():
                shutil.rmtree(oldest_path)
            logger.debug(f"Evicted checkpoint {oldest.checkpoint_id} for agent {agent_id}")

        logger.info(
            f"Saved checkpoint {checkpoint_id} for {agent_id} "
            f"(pool: {len(self._index[agent_id])}/{self._pool_size})"
        )
        return checkpoint_id

    def load(self, agent_id: str, checkpoint_id: str) -> dict:
        """Load state_dict from a checkpoint."""
        ckpt_dir = self._base_dir / agent_id / checkpoint_id
        model_path = ckpt_dir / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        return torch.load(model_path, weights_only=True)

    def load_optimizer(self, agent_id: str, checkpoint_id: str) -> dict | None:
        """Load optimizer state from a checkpoint, if it exists."""
        opt_path = self._base_dir / agent_id / checkpoint_id / "optimizer.pt"
        if opt_path.exists():
            return torch.load(opt_path, weights_only=True)
        return None

    def list_checkpoints(self, agent_id: str) -> list[CheckpointInfo]:
        """List all checkpoints for an agent, ordered by policy_version."""
        return list(self._index.get(agent_id, []))

    def get_latest(self, agent_id: str) -> CheckpointInfo | None:
        """Get the most recent checkpoint for an agent."""
        ckpts = self._index.get(agent_id, [])
        return ckpts[-1] if ckpts else None

    def get_random(self, agent_id: str) -> CheckpointInfo | None:
        """Get a random checkpoint from the pool."""
        ckpts = self._index.get(agent_id, [])
        return random.choice(ckpts) if ckpts else None

    @property
    def agents(self) -> list[str]:
        """List all agents that have checkpoints."""
        return list(self._index.keys())
