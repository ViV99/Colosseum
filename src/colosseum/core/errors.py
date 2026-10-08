"""Exception types raised by Colosseum."""

from __future__ import annotations


class ColosseumError(Exception):
    """Base class for all Colosseum errors."""


class ConfigError(ColosseumError):
    """The configuration is invalid or inconsistent (raised before any process starts)."""


class EnvContractError(ColosseumError):
    """An environment violated the ``BaseEnv`` contract (shapes, masks, flags)."""


class DataError(ConfigError, ValueError):
    """Input data (e.g. offline BC data) is unreadable, has the wrong format or does not fit
    the configured model. Reported at startup like a config error; also a ``ValueError``."""
