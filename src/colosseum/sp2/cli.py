"""CLI entry point (``python -m colosseum.sp2``; the ``colosseum`` console script after the switch)."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import click

# Test hook: when set, the CLI touches this file when ``_interrupts`` becomes active, i.e. from
# then on a Ctrl-C is handled by it (lifecycle tests send SIGINT during startup only after the
# file exists).
_STARTUP_MARKER_ENV = "COLOSSEUM_TEST_STARTUP_MARKER"


@contextmanager
def _interrupts() -> Iterator[None]:
    """Ctrl-C before the run installs its own signal handling (imports, config loading,
    validation): one line on stderr and exit code 130, as after a handled SIGINT (click would
    turn it into "Aborted!" with exit code 1). Wraps every command (``_InterruptibleGroup``)."""
    try:
        marker = os.environ.get(_STARTUP_MARKER_ENV)
        if marker:
            open(marker, "a").close()
        yield
    except KeyboardInterrupt:
        click.echo("Interrupted", err=True)
        sys.exit(130)


class _InterruptibleGroup(click.Group):
    """Runs the group callback and the chosen command inside ``_interrupts``."""

    def invoke(self, ctx: click.Context):
        with _interrupts():
            return super().invoke(ctx)


@click.group(cls=_InterruptibleGroup)
def main() -> None:
    """Colosseum — Distributed RL Training Framework."""
    # User code (e.g. ``examples.*`` or ``my_game.*``) is imported relative to the cwd.
    # Spawned children inherit sys.path.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


@contextmanager
def _config_errors() -> Iterator[None]:
    """A config (or env contract, or input data) problem found at startup: one line on stderr,
    exit code 1, no traceback."""
    from colosseum.core.errors import ConfigError, EnvContractError

    try:
        yield
    except (ConfigError, EnvContractError) as e:
        click.echo(f"Config error: {e}", err=True)
        sys.exit(1)


_SET_HELP_YAML = (
    " Values are YAML scalars/lists (null, true, 1e-4, [1, 2]); quote a value to force a string, "
    "e.g. --set run.name='\"123\"'."
)


def _parse_overrides(overrides: tuple[str, ...]) -> dict:
    """Parse ``--set key=value`` pairs; values use YAML semantics (null, numbers, lists)."""
    from colosseum.sp2.core.config import parse_override_value

    result = {}
    for ov in overrides:
        if "=" not in ov:
            raise click.BadParameter(f"Override must be key=value, got: {ov!r}")
        key, value = ov.split("=", 1)
        key = key.strip()
        result[key] = parse_override_value(value, key=key)
    return result


def _parse_agent_spec(spec: str) -> tuple[str, str]:
    """``name=path`` -> (name, path)."""
    name, sep, path = spec.partition("=")
    if not sep or not name or not path:
        raise click.BadParameter(f"expected name=path, got {spec!r}", param_hint="'--agent'")
    return name, path


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--set", "overrides", multiple=True,
              help="Override config values (e.g., --set rollout.num_workers=8)." + _SET_HELP_YAML)
def train(config: str, overrides: tuple[str, ...]) -> None:
    """Train the config's agents (single machine).

    Exit code: 0 budget reached, 1 config error or a child process died, 130 SIGINT, 143 SIGTERM.
    """
    with _config_errors():
        from colosseum.sp2.launcher import run_training  # imports torch: about a second

        code = run_training(config, overrides=_parse_overrides(overrides) or None)
    sys.exit(code)


if __name__ == "__main__":
    main()
