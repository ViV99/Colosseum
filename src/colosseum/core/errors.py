"""Exception types raised by Colosseum."""

from __future__ import annotations


class ColosseumError(Exception):
    """Base class for all Colosseum errors."""


class ConfigError(ColosseumError):
    """The configuration is invalid or inconsistent (raised before any process starts)."""


class EnvContractError(ColosseumError):
    """An environment violated the ``BaseEnv`` contract (shapes, masks, flags)."""
