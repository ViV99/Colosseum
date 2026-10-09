"""Offline behavioral cloning (BC) on observation and action trees.

Trains a :class:`~colosseum.sp2.networks.model.PolicyModel` by maximum likelihood on recorded
expert decisions of one agent (``colosseum bc --agent A``): the model and the spaces come from
that agent's config and roles.

Data format: one or more ``.pt`` files (``torch.save`` of a dict of tensors) with
- ``observations``: the role's observation tree (a tensor, or a dict of tensors for Dict
  spaces), every leaf ``[N, *leaf_shape]``; dtypes are kept (the model casts);
- ``actions``: the role's action tree with a leading ``[N]`` (``ActionSpec.allocate_actions``
  layout: integers for discrete parts, floats for boxes, ``Units`` groups as ``[N, U, ...]``);
- ``action_masks`` (optional): the mask tree (``ActionSpec.full_mask`` layout) with a leading
  ``[N]``, bool, True = legal;
- ``dones`` (optional): ``[N]`` bool, True when decision t ends its episode.

Loss = minus log-prob of the expert action: the joint log-prob when the action has one decider
(K = 1); with K > 1 (``Units``) the mean of ``unit_log_prob`` over the valid deciders, so a
decision weighs the same whatever its number of units. Decisions without a valid decider are
skipped. The model runs without its value path (``step``, or ``unroll(with_value=False)``).

``add_data`` checks the data against the agent's spaces, independent of the model, and raises
:class:`DataError` (naming the file in ``load_data`` and the first bad decision): tree structure
and shapes, integer dtypes and ranges of discrete values (also those of absent units), finite
float leaves, and the legality of every expert action under its ``action_masks``
(:meth:`ActionSpec.first_illegal_action`: no empty non-units row, the recorded value allowed;
for units, every valid component of a present unit).

Stateless models train on shuffled decisions. Stateful models (``model.is_stateful``) train
with ``unroll`` over contiguous windows of ``seq_len`` decisions, from ``initial_state`` and
with the state reset after every ``done``; a window that starts mid-episode loses the context
before it, and a random window offset per epoch moves those cut points. The last decision of
every file (``add_data`` call) ends an episode; without ``dones`` a file is one episode (with a
warning for stateful models). Observation normalizers are updated once per sample: ``train``
feeds the data added since the previous ``train`` call to ``model.update_normalizers`` before
the first epoch.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import torch

from colosseum.core.errors import DataError
from colosseum.sp2.core.specs import ActionGroup, ActionSpec, ObsSpec
from colosseum.sp2.core.tree import tree_get, tree_index, tree_leaves, tree_map, tree_paths
from colosseum.sp2.networks.dist import Distribution
from colosseum.sp2.networks.model import PolicyModel

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = {"observations", "actions"}
_KNOWN_KEYS = _REQUIRED_KEYS | {"action_masks", "dones"}
_EVAL_BATCH = 4096
# Numerical backstop only (legality is checked in ``add_data``): an NLL this large means a
# (near-)zero probability, e.g. a collapsed std or diverged logits.
_HUGE_NLL = 1e30


def _window_index(n: int, seq_len: int, offset: int = 0) -> torch.Tensor:
    """``[W, seq_len]`` dataset indices of consecutive windows; -1 marks padding.

    The windows tile ``range(n)`` in order: ``[0, offset)`` when ``offset > 0``, then
    ``[offset, offset + seq_len)``, and so on; the last window may be shorter.
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


def per_sample_nll(dist: Distribution, actions: Any, num_deciders: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(nll [B], valid [B] bool)``: joint NLL for K = 1, mean NLL over valid deciders for K > 1."""
    if num_deciders == 1:
        nll = -dist.log_prob(actions)
        return nll, torch.ones_like(nll, dtype=torch.bool)
    unit_lp = dist.unit_log_prob(actions)
    unit_valid = dist.unit_valid(actions)
    count = unit_valid.sum(dim=-1)
    total = torch.where(unit_valid, unit_lp, torch.zeros_like(unit_lp)).sum(dim=-1)
    return -total / count.clamp(min=1).to(total.dtype), count > 0


def _fmt(path: tuple[str, ...]) -> str:
    return "/".join(path) or "<root>"


def _as_tree(name: str, value: Any) -> Any:
    """Dicts stay dicts (string keys), every leaf becomes a tensor; DataError if one cannot."""
    if isinstance(value, dict):
        return {str(k): _as_tree(name, v) for k, v in value.items()}
    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as e:
        raise DataError(f"BC {name}: a leaf of type {type(value).__name__} is not a numeric tensor ({e})") from e


def _leaf_shapes(tree: Any) -> dict[tuple[str, ...], tuple[int, ...]]:
    return {path: tuple(leaf.shape) for path, leaf in zip(tree_paths(tree), tree_leaves(tree), strict=True)}


def _check_tree(name: str, data: Any, expected: Any, n: int) -> None:
    """``data`` must have the paths of ``expected`` (allocated with a leading 1), a leading ``n``
    and the per-leaf trailing shapes of ``expected``."""
    got, want = _leaf_shapes(data), _leaf_shapes(expected)
    if set(got) != set(want):
        raise DataError(f"BC {name} has leaves {sorted(map(_fmt, got))}, but the agent's space needs "
                        f"{sorted(map(_fmt, want))}")
    for path, shape in want.items():
        if got[path][:1] != (n,) or got[path][1:] != shape[1:]:
            raise DataError(f"BC {name} leaf {_fmt(path)} has shape {got[path]}, expected ({n}, *{shape[1:]})")


def _check_finite(name: str, tree: Any) -> None:
    """Float leaves must be finite (DataError naming the leaf and the first bad decision)."""
    for path, leaf in zip(tree_paths(tree), tree_leaves(tree), strict=True):
        if leaf.is_floating_point():
            bad = ~torch.isfinite(leaf).reshape(leaf.shape[0], -1).all(dim=-1)
            if bad.any():
                raise DataError(f"BC {name} leaf {_fmt(path)} has NaN/inf values (first at decision "
                                f"{int(bad.nonzero()[0, 0])})")


def _discrete_ranges(group: ActionGroup) -> list[tuple[tuple[str, ...], int | None, int]]:
    """``(subpath, column, size)`` of every discrete value of ``group``'s action leaf/leaves."""
    if group.kind == "discrete":
        return [((), None, group.nvec[0])]
    if group.kind == "multi_discrete":
        return [((), i, n) for i, n in enumerate(group.nvec)]
    if group.kind != "units":
        return []
    units = group.units
    discrete = [(i, c) for i, c in enumerate(units.components) if c.kind == "discrete"]
    if units.per_unit_kind == "dict":
        return [((c.name,), None, c.size) for _i, c in discrete]
    if units.per_unit_kind == "multi_discrete":
        return [((), i, c.size) for i, c in discrete]
    return [((), None, c.size) for _i, c in discrete]  # discrete per unit (box: none)


def _check_action_values(spec: ActionSpec, actions: Any) -> None:
    """Discrete parts: integer dtype and values in ``[0, n)`` (DataError otherwise)."""
    for group in spec.groups:
        for subpath, column, size in _discrete_ranges(group):
            path = (*group.path, *subpath)
            leaf = tree_get(actions, path)
            where = _fmt(path) + ("" if column is None else f"[..., {column}]")
            if leaf.is_floating_point():
                raise DataError(f"BC actions leaf {_fmt(path)} is floating point but that part of the action "
                                f"space is discrete; store discrete actions as integers")
            values = leaf if column is None else leaf[..., column]
            if values.numel() and (int(values.min()) < 0 or int(values.max()) >= size):
                raise DataError(f"BC actions {where} has values outside [0, {size}) (min {int(values.min())}, "
                                f"max {int(values.max())})")


def _has_accuracy(spec: ActionSpec) -> bool:
    """Accuracy is reported for actions with a discrete part and without ``Units`` groups."""
    return not spec.has_units and any(g.kind in ("discrete", "multi_discrete") for g in spec.groups)


def _discrete_hits(spec: ActionSpec, mode: Any, actions: Any) -> torch.Tensor:
    """1.0 where the mode equals the expert on every discrete group (``_has_accuracy(spec)`` holds)."""
    hits = None
    for group in spec.groups:
        if group.kind not in ("discrete", "multi_discrete"):
            continue
        same = tree_get(mode, group.path) == tree_get(actions, group.path)
        if same.dim() > 1:
            same = same.all(dim=-1)
        hits = same if hits is None else hits & same
    return hits.float()


class OfflineBCTrainer:
    """Supervised behavioral cloning from offline data (module docstring).

    Usage::

        trainer = OfflineBCTrainer(model, action_spec, obs_spec, lr=1e-3, seq_len=64)
        trainer.load_data("expert_games/")      # or add_data(obs_tree, action_tree, mask_tree, dones)
        metrics = trainer.train(num_epochs=10, batch_size=256)
    """

    def __init__(self, model: PolicyModel, action_spec: ActionSpec, obs_spec: ObsSpec, lr: float = 1e-3,
                 device: str | torch.device = "cpu", seq_len: int = 64) -> None:
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        self._device = torch.device(device)
        self._model = model.to(self._device)
        self._action_spec = action_spec
        self._obs_spec = obs_spec
        self._num_deciders = int(action_spec.num_deciders)
        self._seq_len = int(seq_len)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        self._observations: list[Any] = []
        self._actions: list[Any] = []
        self._masks: list[Any] = []
        self._dones: list[torch.Tensor] = []
        self._normalized_upto = 0  # decisions already fed to model.update_normalizers

    @property
    def model(self) -> PolicyModel:
        return self._model

    @property
    def num_samples(self) -> int:
        return sum(int(d.shape[0]) for d in self._dones)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def add_data(self, observations: Any, actions: Any, action_masks: Any = None, dones: Any = None) -> None:
        """Add ``N`` expert decisions (trees with a leading ``N``; module docstring)."""
        obs = _as_tree("observations", observations)
        acts = _as_tree("actions", actions)
        leaves = tree_leaves(obs)
        n = int(leaves[0].shape[0]) if leaves and leaves[0].dim() > 0 else 0
        if n == 0:
            raise DataError("BC data is empty (no observations)")
        _check_tree("observations", obs, self._obs_spec.allocate((1,)), n)
        _check_tree("actions", acts, self._action_spec.allocate_actions((1,)), n)
        _check_action_values(self._action_spec, acts)
        _check_finite("observations", obs)
        _check_finite("actions", acts)
        acts = tree_map(lambda x: x.float() if x.is_floating_point() else x.long(), acts)
        masks = None
        if action_masks is not None:
            if not self._action_spec.has_masks:
                raise DataError("BC data has action_masks but the agent's action space has nothing to mask; "
                                "drop 'action_masks' from the data")
            masks = tree_map(lambda x: x.bool(), _as_tree("action_masks", action_masks))
            _check_tree("action_masks", masks, self._action_spec.full_mask((1,)), n)
            illegal = self._action_spec.first_illegal_action(acts, masks)
            if illegal is not None:
                index, reason = illegal
                raise DataError(f"BC decision {index}: {reason}; expert actions must be legal under their "
                                f"action_masks (fix the data)")
        if self._masks and (masks is None) != (self._masks[0] is None):
            raise DataError("BC data: either every batch/file has 'action_masks' or none does")
        if dones is None:
            if self._model.is_stateful:
                logger.warning("BC data has no 'dones': treating these %d decisions as one episode "
                               "(this matters only for stateful models, which this one is)", n)
            done_t = torch.zeros(n, dtype=torch.bool)
        else:
            done_t = _as_tree("dones", dones).bool().reshape(-1).clone()
            if done_t.shape[0] != n:
                raise DataError(f"BC data: {n} observations but {done_t.shape[0]} dones")
        done_t[-1] = True  # never carry state across add_data calls (separate files are unrelated)
        self._observations.append(obs)
        self._actions.append(acts)
        self._masks.append(masks)
        self._dones.append(done_t)

    def load_data(self, path: str | Path) -> int:
        """Load one ``.pt`` file or every ``*.pt`` in a directory; returns the decisions loaded."""
        path = Path(path)
        files = sorted(path.glob("*.pt")) if path.is_dir() else [path]
        if not files:
            raise DataError(f"no .pt files in {path}")
        total = 0
        for f in files:
            try:
                data = torch.load(f, map_location="cpu", weights_only=True)
            except OSError as e:  # PermissionError, IsADirectoryError, FileNotFoundError, I/O errors
                reason = e.strerror or str(e) or type(e).__name__
                raise DataError(f"{f}: cannot read the BC data file ({type(e).__name__}: {reason})") from e
            except (EOFError, RuntimeError, pickle.UnpicklingError) as e:
                detail = (str(e).strip().splitlines() or [""])[0]
                raise DataError(f"{f}: not a readable torch.save file ({type(e).__name__}: {detail})") from e
            if not isinstance(data, dict):
                raise DataError(f"{f}: expected a dict with keys {sorted(_KNOWN_KEYS)}")
            missing = _REQUIRED_KEYS - set(data)
            if missing:
                raise DataError(f"{f}: missing BC data keys {sorted(missing)} (need observations, actions)")
            unknown = set(data) - _KNOWN_KEYS
            if unknown:
                logger.warning("%s: ignoring unknown BC data keys %s", f, sorted(unknown))
            before = self.num_samples
            try:
                self.add_data(data["observations"], data["actions"], data.get("action_masks"), data.get("dones"))
            except DataError as e:
                raise DataError(f"{f}: {e}") from e
            loaded = self.num_samples - before
            total += loaded
            logger.info("Loaded %d BC decisions from %s", loaded, f)
        logger.info("BC dataset: %d decisions", self.num_samples)
        return total

    def _dataset(self) -> tuple[Any, Any, Any, torch.Tensor]:
        def cat(trees: list[Any]) -> Any:
            return tree_map(lambda *xs: torch.cat(xs), *trees)

        masks = None if self._masks[0] is None else cat(self._masks)
        return cat(self._observations), cat(self._actions), masks, torch.cat(self._dones)

    def _to_device(self, tree: Any) -> Any:
        return None if tree is None else tree_map(lambda x: x.to(self._device), tree)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, num_epochs: int = 10, batch_size: int = 256, log_interval: int = 1) -> dict[str, float]:
        """Run BC; returns the final-epoch NLL ``bc_loss`` and, for discrete actions, ``accuracy``."""
        if not self._observations:
            raise ValueError("No training data. Call add_data() or load_data() first.")
        if num_epochs < 1 or batch_size < 1:
            raise ValueError("num_epochs and batch_size must be >= 1")
        obs, actions, masks, dones = self._dataset()
        n = int(dones.shape[0])
        self._update_normalizers(obs, n)
        stateful = self._model.is_stateful
        logger.info("BC training: %d decisions, %d epochs, batch_size=%d, %s", n, num_epochs, batch_size,
                    f"stateful (seq_len={self._seq_len})" if stateful else "stateless")
        self._model.train()
        epoch_losses: list[float] = []
        for epoch in range(num_epochs):
            if stateful:
                loss = self._sequence_epoch(obs, actions, masks, dones, batch_size)
            else:
                loss = self._flat_epoch(obs, actions, masks, n, batch_size)
            epoch_losses.append(loss)
            if (epoch + 1) % log_interval == 0:
                logger.info("BC epoch %d/%d: nll=%.4f", epoch + 1, num_epochs, loss)
        metrics = {"bc_loss": epoch_losses[-1], "bc_loss_first_epoch": epoch_losses[0],
                   "num_epochs": float(num_epochs), "num_samples": float(n)}
        accuracy = self._accuracy(obs, actions, masks, dones, n)
        if accuracy is not None:
            metrics["accuracy"] = accuracy
        return metrics

    def _update_normalizers(self, obs: Any, n: int) -> None:
        """Feed every decision added since the last call to ``model.update_normalizers`` once."""
        for start in range(self._normalized_upto, n, _EVAL_BATCH):
            idx = torch.arange(start, min(start + _EVAL_BATCH, n))
            self._model.update_normalizers(self._to_device(tree_index(obs, idx)))
        self._normalized_upto = n

    def _weighted_nll(self, dist: Distribution, actions: Any,
                      weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nll, valid = per_sample_nll(dist, actions, self._num_deciders)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        active = weights > 0
        nan = torch.isnan(nll) & active
        if nan.any():
            raise RuntimeError(
                f"BC NLL is NaN for {int(nan.sum())} samples: training diverged or the data contains "
                f"NaN/inf (check observations/actions, lower the learning rate)"
            )
        huge = (nll > _HUGE_NLL) & active
        if huge.any():
            raise RuntimeError(
                f"BC NLL exceeds {_HUGE_NLL:g} for {int(huge.sum())} samples: the policy gives the expert "
                f"actions (near-)zero probability (a collapsed std or diverged logits; lower the learning rate)"
            )
        nll = torch.where(active, nll, torch.zeros_like(nll))
        return (nll * weights).sum(), weights.sum()

    def _step_dist(self, obs: Any, masks: Any, batch: int) -> Distribution:
        state0 = self._model.initial_state(batch, self._device)
        return self._model.step(self._to_device(obs), state0, self._to_device(masks)).dist

    def _optimize(self, loss_sum: torch.Tensor, weight: torch.Tensor) -> None:
        if float(weight) > 0:  # a batch without a valid decider has nothing to learn
            self._optimizer.zero_grad()
            (loss_sum / weight).backward()
            self._optimizer.step()

    def _flat_epoch(self, obs: Any, actions: Any, masks: Any, n: int, batch_size: int) -> float:
        total, count = 0.0, 0.0
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            dist = self._step_dist(tree_index(obs, idx), None if masks is None else tree_index(masks, idx), len(idx))
            weights = torch.ones(len(idx), device=self._device)
            loss_sum, weight = self._weighted_nll(dist, self._to_device(tree_index(actions, idx)), weights)
            self._optimize(loss_sum, weight)
            total += float(loss_sum.detach())
            count += float(weight)
        return total / max(count, 1.0)

    def _sequence_forward(self, index: torch.Tensor, obs: Any, masks: Any,
                          dones: torch.Tensor) -> tuple[Distribution, torch.Tensor]:
        """Unroll windows ``index [L, b]``; returns (dist over L*b time-major rows, weights)."""
        valid = index >= 0
        safe = index.clamp(min=0)
        o = self._to_device(tree_index(obs, safe))                          # leaves [L, b, ...]
        reset_after = (dones[safe] | ~valid).to(self._device)               # [L, b]
        m = None if masks is None else self._to_device(tree_index(masks, safe))
        state0 = self._model.initial_state(index.shape[1], self._device)
        dist = self._model.unroll(o, state0, reset_after, m, with_value=False).dist
        return dist, valid.reshape(-1).to(self._device).float()

    def _window_actions(self, actions: Any, index: torch.Tensor) -> Any:
        """Actions of the windows ``index [L, b]``, time-major flattened to ``[L*b]`` (``unroll``'s order)."""
        return self._to_device(tree_index(actions, index.clamp(min=0).reshape(-1)))

    def _sequence_epoch(self, obs: Any, actions: Any, masks: Any, dones: torch.Tensor, batch_size: int) -> float:
        seq_len = self._seq_len
        offset = int(torch.randint(0, seq_len, (1,)).item())
        windows = _window_index(int(dones.shape[0]), seq_len, offset)
        per_batch = max(1, batch_size // seq_len)
        order = torch.randperm(len(windows))
        total, count = 0.0, 0.0
        for start in range(0, len(windows), per_batch):
            index = windows[order[start:start + per_batch]].t()          # [L, b]
            dist, weights = self._sequence_forward(index, obs, masks, dones)
            loss_sum, weight = self._weighted_nll(dist, self._window_actions(actions, index), weights)
            self._optimize(loss_sum, weight)
            total += float(loss_sum.detach())
            count += float(weight)
        return total / max(count, 1.0)

    @torch.no_grad()
    def _accuracy(self, obs: Any, actions: Any, masks: Any, dones: torch.Tensor, n: int) -> float | None:
        if not _has_accuracy(self._action_spec):
            return None
        self._model.eval()
        try:
            correct, total = 0.0, 0.0
            if self._model.is_stateful:
                windows = _window_index(n, self._seq_len, 0)
                per_batch = max(1, _EVAL_BATCH // self._seq_len)
                for start in range(0, len(windows), per_batch):
                    index = windows[start:start + per_batch].t()
                    dist, weights = self._sequence_forward(index, obs, masks, dones)
                    hits = _discrete_hits(self._action_spec, dist.mode(), self._window_actions(actions, index))
                    correct += float((hits * weights).sum())
                    total += float(weights.sum())
            else:
                for start in range(0, n, _EVAL_BATCH):
                    idx = torch.arange(start, min(start + _EVAL_BATCH, n))
                    dist = self._step_dist(tree_index(obs, idx), None if masks is None else tree_index(masks, idx),
                                           len(idx))
                    hits = _discrete_hits(self._action_spec, dist.mode(), self._to_device(tree_index(actions, idx)))
                    correct += float(hits.sum())
                    total += float(len(idx))
            return correct / max(total, 1.0)
        finally:
            self._model.train()
