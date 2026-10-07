"""ActionSpec: codec for composite action spaces.

Converts between gymnasium composite spaces (Dict, Tuple, MultiDiscrete) and
flat numpy/torch tensors used internally by the training pipeline.

For simple spaces (Discrete, Box) the codec is a no-op pass-through.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import numpy as np

_MAX_DISCRETE = 2**24  # float32 exact-integer limit


@dataclass(frozen=True)
class ActionComponent:
    """One sub-space within a composite action."""

    name: str
    offset: int          # index in flat action vector
    size: int            # 1 for Discrete, D for Box(D,)
    is_discrete: bool
    num_categories: int  # Discrete(N) → N; else 0
    mask_offset: int     # index in flat mask vector
    mask_size: int       # Discrete(N) → N; else 0


@dataclass(frozen=True)
class ActionSpec:
    """Describes the flat representation of an action space.

    For simple spaces (Discrete, Box), ``is_composite`` is False and all
    pipeline code uses existing fast paths with zero overhead.

    For composite spaces (Dict, Tuple, MultiDiscrete), actions are stored as
    flat ``float32`` vectors of length ``flat_size``.  The ``decode``/``encode``
    methods convert between flat and structured representations at the env
    boundary.
    """

    components: tuple[ActionComponent, ...]
    flat_size: int
    flat_mask_size: int
    action_shape: tuple       # () for Discrete, (D,) for Box, (flat_size,) for composite
    numpy_dtype: type         # np.int64 for simple Discrete, np.float32 otherwise
    is_composite: bool
    space_type: str           # "discrete", "box", "dict", "tuple", "multi_discrete"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_space(cls, space) -> ActionSpec:
        """Auto-derive an ActionSpec from a gymnasium Space."""
        import gymnasium

        if isinstance(space, gymnasium.spaces.Discrete):
            if space.n > _MAX_DISCRETE:
                raise ValueError(
                    f"Discrete({space.n}) exceeds float32 exact-integer limit "
                    f"({_MAX_DISCRETE}).  Composite encoding would lose precision."
                )
            comp = ActionComponent(
                name="_scalar", offset=0, size=1,
                is_discrete=True, num_categories=int(space.n),
                mask_offset=0, mask_size=int(space.n),
            )
            return cls(
                components=(comp,),
                flat_size=1,
                flat_mask_size=int(space.n),
                action_shape=(),
                numpy_dtype=np.int64,
                is_composite=False,
                space_type="discrete",
            )

        if isinstance(space, gymnasium.spaces.Box):
            flat = int(np.prod(space.shape))
            comp = ActionComponent(
                name="_vector", offset=0, size=flat,
                is_discrete=False, num_categories=0,
                mask_offset=0, mask_size=0,
            )
            return cls(
                components=(comp,),
                flat_size=flat,
                flat_mask_size=0,
                action_shape=space.shape,
                numpy_dtype=np.float32,
                is_composite=False,
                space_type="box",
            )

        if isinstance(space, gymnasium.spaces.Dict):
            return cls._from_named_spaces(
                {k: space.spaces[k] for k in sorted(space.spaces.keys())},
                space_type="dict",
            )

        if isinstance(space, gymnasium.spaces.Tuple):
            named = {str(i): s for i, s in enumerate(space.spaces)}
            return cls._from_named_spaces(named, space_type="tuple")

        if isinstance(space, gymnasium.spaces.MultiDiscrete):
            named = {str(i): gymnasium.spaces.Discrete(int(n))
                     for i, n in enumerate(space.nvec)}
            return cls._from_named_spaces(named, space_type="multi_discrete")

        raise TypeError(f"Unsupported action space type: {type(space).__name__}")

    @classmethod
    def _from_named_spaces(
        cls,
        named: dict[str, Any],
        space_type: str,
    ) -> ActionSpec:
        """Build a composite ActionSpec from an ordered dict of sub-spaces."""
        import gymnasium

        components: list[ActionComponent] = []
        offset = 0
        mask_offset = 0

        for name in sorted(named.keys()):
            sub = named[name]

            # Only allow flat (non-composite) sub-spaces
            if isinstance(sub, (gymnasium.spaces.Dict,
                                gymnasium.spaces.Tuple,
                                gymnasium.spaces.MultiDiscrete)):
                raise ValueError(
                    f"Nested composite spaces are not supported (key={name!r}). "
                    f"Flatten your space to a single-level Dict/Tuple."
                )

            if isinstance(sub, gymnasium.spaces.Discrete):
                if sub.n > _MAX_DISCRETE:
                    raise ValueError(
                        f"Discrete({sub.n}) in key {name!r} exceeds float32 "
                        f"exact-integer limit ({_MAX_DISCRETE})."
                    )
                components.append(ActionComponent(
                    name=name, offset=offset, size=1,
                    is_discrete=True, num_categories=int(sub.n),
                    mask_offset=mask_offset, mask_size=int(sub.n),
                ))
                offset += 1
                mask_offset += int(sub.n)

            elif isinstance(sub, gymnasium.spaces.Box):
                flat = int(np.prod(sub.shape))
                components.append(ActionComponent(
                    name=name, offset=offset, size=flat,
                    is_discrete=False, num_categories=0,
                    mask_offset=mask_offset, mask_size=0,
                ))
                offset += flat
                # mask_offset unchanged — continuous has no mask

            else:
                raise TypeError(
                    f"Unsupported sub-space type for key {name!r}: "
                    f"{type(sub).__name__}. Use Discrete or Box."
                )

        return cls(
            components=tuple(components),
            flat_size=offset,
            flat_mask_size=mask_offset,
            action_shape=(offset,),
            numpy_dtype=np.float32,
            is_composite=True,
            space_type=space_type,
        )

    # ------------------------------------------------------------------
    # Codec
    # ------------------------------------------------------------------

    def decode(self, flat: np.ndarray) -> Any:
        """Flat array → structured action for ``env.step()``.

        For non-composite specs, returns the input unchanged.
        """
        if not self.is_composite:
            return flat  # pass-through: scalar (Discrete) or array (Box)

        if self.space_type == "multi_discrete":
            # Return a flat int array like gymnasium.MultiDiscrete.sample()
            return np.array(
                [int(flat[c.offset]) for c in self.components],
                dtype=np.int64,
            )

        if self.space_type == "tuple":
            result = []
            for c in self.components:
                if c.is_discrete:
                    result.append(int(flat[c.offset]))
                else:
                    result.append(flat[c.offset:c.offset + c.size].copy())
            return tuple(result)

        # Dict (default for composite)
        result = {}
        for c in self.components:
            if c.is_discrete:
                result[c.name] = int(flat[c.offset])
            else:
                val = flat[c.offset:c.offset + c.size]
                result[c.name] = val.copy() if val.size > 1 else float(val[0])
        return result

    def encode(self, structured: Any) -> np.ndarray:
        """Structured action → flat float32 array.  For testing/utility."""
        if not self.is_composite:
            return np.asarray(structured, dtype=self.numpy_dtype)

        flat = np.zeros(self.flat_size, dtype=np.float32)

        if self.space_type == "multi_discrete":
            arr = np.asarray(structured)
            for i, c in enumerate(self.components):
                flat[c.offset] = float(arr[i])
            return flat

        if self.space_type == "tuple":
            for c, val in zip(self.components, structured):
                if c.is_discrete:
                    flat[c.offset] = float(val)
                else:
                    flat[c.offset:c.offset + c.size] = np.asarray(val, dtype=np.float32).flat
            return flat

        # Dict
        for c in self.components:
            val = structured[c.name]
            if c.is_discrete:
                flat[c.offset] = float(val)
            else:
                flat[c.offset:c.offset + c.size] = np.asarray(val, dtype=np.float32).flat
        return flat

    def flatten_mask(self, mask: Union[np.ndarray, dict]) -> np.ndarray:
        """Normalize an action mask to a flat bool array of ``flat_mask_size``.

        Accepts:
          - ``np.ndarray`` of shape ``(flat_mask_size,)`` — pass-through.
          - ``dict[str, np.ndarray]`` — assemble from per-component masks.
            Missing keys default to all-True (all actions valid).
        """
        if isinstance(mask, np.ndarray):
            return mask.astype(bool)

        if not isinstance(mask, dict):
            raise TypeError(
                f"Expected np.ndarray or dict for action mask, got {type(mask).__name__}"
            )

        flat = np.ones(self.flat_mask_size, dtype=bool)
        for c in self.components:
            if c.mask_size == 0:
                continue  # continuous — no mask
            if c.name in mask:
                flat[c.mask_offset:c.mask_offset + c.mask_size] = np.asarray(
                    mask[c.name], dtype=bool,
                )
        return flat
