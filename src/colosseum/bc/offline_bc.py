"""Offline behavioral cloning (BC).

Trains a :class:`~colosseum.networks.model.PolicyModel` by maximum likelihood on
recorded expert transitions: loss = -log pi(a | s) for every distribution type
(Categorical, DiagGaussian, Composite), with the recorded action masks applied.

Data format: one or more ``.pt`` files (``torch.save`` of a dict) with
- ``observations``: ``[N, *obs_shape]`` (any numeric dtype; cast to float32);
- ``actions``: ``[N]`` integer tensor for a Discrete space, ``[N, D]`` float for
  a Box, ``[N, flat_size]`` float in the ``ActionSpec`` flat layout for
  Dict/Tuple/MultiDiscrete spaces;
- ``action_masks`` (optional): ``[N, flat_mask_size]`` bool, True = legal;
- ``dones`` (optional): ``[N]`` bool, True when transition t ends its episode.

Stateless models train on shuffled transitions. Stateful models
(``model.is_stateful``) train with ``model.unroll`` over contiguous windows of
``seq_len`` transitions. Each window starts from ``model.initial_state`` and the
state is reset after every ``done``. A window that starts mid-episode loses the
context before it (``UnrollOutput`` carries no final state); a random window
offset per epoch moves those cut points. Without ``dones``, the transitions of
each file (``add_data`` call) are treated as one episode, with a warning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from colosseum.networks.distributions import CategoricalDist, CompositeDist, Distribution
from colosseum.networks.model import PolicyModel

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = {"observations", "actions"}
_KNOWN_KEYS = _REQUIRED_KEYS | {"action_masks", "dones"}
_EVAL_BATCH = 4096


def _window_index(n: int, seq_len: int, offset: int = 0) -> torch.Tensor:
    """``[W, seq_len]`` dataset indices of consecutive windows; -1 marks padding.

    The windows tile ``range(n)`` in order: ``[0, offset)`` when ``offset > 0``,
    then ``[offset, offset + seq_len)``, and so on; the last window may be shorter.
    """
    if n <= 0:
        return torch.empty(0, seq_len, dtype=torch.long)
    starts = list(range(offset, n, seq_len)) if offset > 0 else list(range(0, n, seq_len))
    if offset > 0:
        starts = [0] + starts
    ends = starts[1:] + [n]
    index = torch.full((len(starts), seq_len), -1, dtype=torch.long)
    for w, (start, end) in enumerate(zip(starts, ends)):
        index[w, : end - start] = torch.arange(start, end)
    return index


def _discrete_columns(dist: CompositeDist) -> list[int]:
    return [offset for _name, offset, _size, discrete in dist.components if discrete]


def _hits(dist: Distribution, actions: torch.Tensor) -> torch.Tensor | None:
    """1.0 where the mode matches the expert on every discrete component; None if none."""
    if isinstance(dist, CategoricalDist):
        return (dist.mode() == actions).float()
    if isinstance(dist, CompositeDist):
        cols = _discrete_columns(dist)
        if not cols:
            return None
        return (dist.mode()[:, cols] == actions[:, cols]).all(dim=-1).float()
    return None


class OfflineBCTrainer:
    """Supervised behavioral cloning from offline data.

    Usage::

        trainer = OfflineBCTrainer(build_model(cfg), lr=1e-3, seq_len=64)
        trainer.load_data("expert_games/")      # or add_data(obs, actions, masks, dones)
        metrics = trainer.train(num_epochs=10, batch_size=256)
    """

    def __init__(
        self,
        model: PolicyModel,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
        seq_len: int = 64,
    ) -> None:
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        self._device = torch.device(device)
        self._model = model.to(self._device)
        self._seq_len = int(seq_len)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        self._observations: list[torch.Tensor] = []
        self._actions: list[torch.Tensor] = []
        self._masks: list[torch.Tensor | None] = []
        self._dones: list[torch.Tensor] = []

    @property
    def model(self) -> PolicyModel:
        return self._model

    @property
    def num_samples(self) -> int:
        return sum(len(o) for o in self._observations)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def add_data(
        self,
        observations: Any,
        actions: Any,
        action_masks: Any = None,
        dones: Any = None,
    ) -> None:
        """Add ``N`` transitions (see the module docstring for shapes and dtypes)."""
        obs = torch.as_tensor(observations)
        acts = torch.as_tensor(actions)
        n = obs.shape[0]
        if n == 0:
            raise ValueError("BC data is empty")
        if acts.shape[0] != n:
            raise ValueError(f"BC data: {n} observations but {acts.shape[0]} actions")
        masks = None
        if action_masks is not None:
            masks = torch.as_tensor(action_masks).bool()
            if masks.dim() != 2 or masks.shape[0] != n:
                raise ValueError(f"BC action_masks must have shape [N, mask_size], got {tuple(masks.shape)}")
        if self._masks and (masks is None) != (self._masks[0] is None):
            raise ValueError("BC data: either every batch/file has 'action_masks' or none does")
        if dones is None:
            logger.warning("BC data has no 'dones': treating these %d transitions as one episode", n)
            done_t = torch.zeros(n, dtype=torch.bool)
            done_t[-1] = True
        else:
            done_t = torch.as_tensor(dones).bool().reshape(-1)
            if done_t.shape[0] != n:
                raise ValueError(f"BC data: {n} observations but {done_t.shape[0]} dones")
        self._observations.append(obs)
        self._actions.append(acts)
        self._masks.append(masks)
        self._dones.append(done_t)

    def load_data(self, path: str | Path) -> int:
        """Load one ``.pt`` file or every ``*.pt`` in a directory; returns samples loaded."""
        path = Path(path)
        files = sorted(path.glob("*.pt")) if path.is_dir() else [path]
        if not files:
            raise FileNotFoundError(f"no .pt files in {path}")
        total = 0
        for f in files:
            data = torch.load(f, map_location="cpu", weights_only=True)
            if not isinstance(data, dict):
                raise ValueError(f"{f}: expected a dict with keys {sorted(_KNOWN_KEYS)}")
            missing = _REQUIRED_KEYS - set(data)
            if missing:
                raise ValueError(f"{f}: missing BC data keys {sorted(missing)} (need observations, actions)")
            unknown = set(data) - _KNOWN_KEYS
            if unknown:
                logger.warning("%s: ignoring unknown BC data keys %s", f, sorted(unknown))
            self.add_data(data["observations"], data["actions"], data.get("action_masks"), data.get("dones"))
            n = len(data["observations"])
            total += n
            logger.info("Loaded %d BC samples from %s", n, f)
        logger.info("BC dataset: %d samples", self.num_samples)
        return total

    def _dataset(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        obs = torch.cat(self._observations).float()
        actions = torch.cat(self._actions)
        masks = None if self._masks[0] is None else torch.cat(self._masks)
        dones = torch.cat(self._dones)
        return obs, actions, masks, dones

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, num_epochs: int = 10, batch_size: int = 256, log_interval: int = 1) -> dict[str, float]:
        """Run BC; returns final-epoch NLL ``bc_loss`` and, for discrete parts, ``accuracy``."""
        if not self._observations:
            raise ValueError("No training data. Call add_data() or load_data() first.")
        if num_epochs < 1 or batch_size < 1:
            raise ValueError("num_epochs and batch_size must be >= 1")
        obs, actions, masks, dones = self._dataset()
        actions = self._prepare_actions(obs, actions, masks)
        stateful = self._model.is_stateful
        logger.info(
            "BC training: %d samples, %d epochs, batch_size=%d, %s",
            len(obs), num_epochs, batch_size,
            f"stateful (seq_len={self._seq_len})" if stateful else "stateless",
        )
        self._model.train()
        epoch_losses: list[float] = []
        for epoch in range(num_epochs):
            if stateful:
                loss = self._sequence_epoch(obs, actions, masks, dones, batch_size)
            else:
                loss = self._flat_epoch(obs, actions, masks, batch_size)
            epoch_losses.append(loss)
            if (epoch + 1) % log_interval == 0:
                logger.info("BC epoch %d/%d: nll=%.4f", epoch + 1, num_epochs, loss)
        metrics = {
            "bc_loss": epoch_losses[-1],
            "bc_loss_first_epoch": epoch_losses[0],
            "num_epochs": float(num_epochs),
            "num_samples": float(len(obs)),
        }
        accuracy = self._accuracy(obs, actions, masks, dones)
        if accuracy is not None:
            metrics["accuracy"] = accuracy
        return metrics

    def _prepare_actions(
        self, obs: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor | None,
    ) -> torch.Tensor:
        """Check the action dtype/shape against the policy's distribution; return training actions."""
        with torch.no_grad():
            m = None if masks is None else masks[:1].to(self._device)
            dist = self._model.step(
                obs[:1].to(self._device), self._model.initial_state(1, self._device), m,
            ).dist
        if isinstance(dist, CategoricalDist):
            if actions.is_floating_point():
                raise ValueError(
                    "BC actions are floating point but the policy's action space is discrete "
                    "(CategoricalDist); store discrete actions as an integer tensor of shape [N]"
                )
            if actions.dim() != 1:
                raise ValueError(f"discrete BC actions must have shape [N], got {tuple(actions.shape)}")
            return actions.long()
        actions = actions.float()
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        try:
            action_dim = dist.action_dim
        except NotImplementedError:
            action_dim = None
        if action_dim is not None and (actions.dim() != 2 or actions.shape[1] != action_dim):
            raise ValueError(
                f"BC actions must have shape [N, {action_dim}] for {type(dist).__name__}, "
                f"got {tuple(actions.shape)}"
            )
        if isinstance(dist, CompositeDist):
            cols = _discrete_columns(dist)
            if cols:
                sub = actions[:, cols]
                if not torch.equal(sub, sub.round()):
                    raise ValueError(
                        f"BC actions have non-integer values in the discrete components "
                        f"(flat columns {cols}) of a composite action space"
                    )
        return actions

    @staticmethod
    def _weighted_nll(
        dist: Distribution, actions: torch.Tensor, weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nll = -dist.log_prob(actions)
        bad = ~torch.isfinite(nll) & (weights > 0)
        if bad.any():
            raise ValueError(
                f"{int(bad.sum())} BC expert actions have zero probability under the policy: "
                f"they are illegal under their action_masks (fix the data)"
            )
        nll = torch.where(weights > 0, nll, torch.zeros_like(nll))
        return (nll * weights).sum(), weights.sum()

    def _flat_epoch(
        self, obs: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor | None, batch_size: int,
    ) -> float:
        total, count = 0.0, 0.0
        perm = torch.randperm(len(obs))
        for start in range(0, len(obs), batch_size):
            idx = perm[start:start + batch_size]
            o = obs[idx].to(self._device)
            m = None if masks is None else masks[idx].to(self._device)
            dist = self._model.step(o, self._model.initial_state(len(idx), self._device), m).dist
            weights = torch.ones(len(idx), device=self._device)
            loss_sum, weight = self._weighted_nll(dist, actions[idx].to(self._device), weights)
            self._optimizer.zero_grad()
            (loss_sum / weight).backward()
            self._optimizer.step()
            total += float(loss_sum.detach())
            count += float(weight)
        return total / max(count, 1.0)

    def _sequence_forward(
        self,
        index: torch.Tensor,
        obs: torch.Tensor,
        masks: torch.Tensor | None,
        dones: torch.Tensor,
    ) -> tuple[Distribution, torch.Tensor]:
        """Unroll windows ``index [L, b]``; returns (dist over L*b time-major rows, weights)."""
        valid = index >= 0
        safe = index.clamp(min=0)
        o = obs[safe].to(self._device)                                   # [L, b, *obs]
        d = (dones[safe] | ~valid).to(self._device)                      # [L, b]
        m = None if masks is None else masks[safe].to(self._device)      # [L, b, A]
        state0 = self._model.initial_state(index.shape[1], self._device)
        dist = self._model.unroll(o, state0, d, m).dist
        return dist, valid.reshape(-1).to(self._device).float()

    def _sequence_epoch(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor | None,
        dones: torch.Tensor,
        batch_size: int,
    ) -> float:
        seq_len = self._seq_len
        offset = int(torch.randint(0, seq_len, (1,)).item())
        windows = _window_index(len(obs), seq_len, offset)
        per_batch = max(1, batch_size // seq_len)
        order = torch.randperm(len(windows))
        total, count = 0.0, 0.0
        for start in range(0, len(windows), per_batch):
            index = windows[order[start:start + per_batch]].t()          # [L, b]
            dist, weights = self._sequence_forward(index, obs, masks, dones)
            a = actions[index.clamp(min=0)].reshape(index.numel(), *actions.shape[1:]).to(self._device)
            loss_sum, weight = self._weighted_nll(dist, a, weights)
            self._optimizer.zero_grad()
            (loss_sum / weight).backward()
            self._optimizer.step()
            total += float(loss_sum.detach())
            count += float(weight)
        return total / max(count, 1.0)

    @torch.no_grad()
    def _accuracy(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor | None,
        dones: torch.Tensor,
    ) -> float | None:
        self._model.eval()
        try:
            correct, total = 0.0, 0.0
            if self._model.is_stateful:
                windows = _window_index(len(obs), self._seq_len, 0)
                per_batch = max(1, _EVAL_BATCH // self._seq_len)
                for start in range(0, len(windows), per_batch):
                    index = windows[start:start + per_batch].t()
                    dist, weights = self._sequence_forward(index, obs, masks, dones)
                    a = actions[index.clamp(min=0)].reshape(index.numel(), *actions.shape[1:]).to(self._device)
                    hits = _hits(dist, a)
                    if hits is None:
                        return None
                    correct += float((hits * weights).sum())
                    total += float(weights.sum())
            else:
                for start in range(0, len(obs), _EVAL_BATCH):
                    sl = slice(start, start + _EVAL_BATCH)
                    o = obs[sl].to(self._device)
                    m = None if masks is None else masks[sl].to(self._device)
                    dist = self._model.step(o, self._model.initial_state(len(o), self._device), m).dist
                    hits = _hits(dist, actions[sl].to(self._device))
                    if hits is None:
                        return None
                    correct += float(hits.sum())
                    total += float(len(o))
            return correct / max(total, 1.0)
        finally:
            self._model.train()
