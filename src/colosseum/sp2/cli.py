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


@main.command("validate")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--set", "overrides", multiple=True,
              help="Override config values (e.g., --set env.max_idle_steps=200)." + _SET_HELP_YAML)
def validate_cmd(config: str, overrides: tuple[str, ...]) -> None:
    """Validate a config: GameSpec, roles, matchmaking, env steps under the contract, every agent's model."""
    with _config_errors():
        from colosseum.sp2.core.config import load_config
        from colosseum.sp2.core.registry import validate_config

        cfg = load_config(config, _parse_overrides(overrides) or None)
        validate_config(cfg)
        for aid in cfg.get_trainable_agent_ids():
            click.echo(f"  OK: agent '{aid}'")
    click.echo("Config is valid.")


@main.command("eval")
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Config YAML: its env is used for every match; agents.<name> / networks build .pt agents")
@click.option("--agent", "-a", "agents", required=True, multiple=True,
              help="name=path. path is a checkpoint dir (architecture and roles from its meta.json) or a "
                   ".pt state_dict (architecture and roles of agents.<name>, else the global networks "
                   "playing every role). Repeatable.")
@click.option("--layout", "layouts", multiple=True,
              help="Layout to evaluate (repeatable). Default: every layout the agents can fill.")
@click.option("--num-matches", "-n", default=100, type=click.IntRange(min=1), show_default=True,
              help="Matches per agent pair and layout (one agent or one team: per agent; cross-play: per "
                   "composition). With several agents and a layout of two or more teams an odd count is "
                   "rounded up, so every agent plays every side equally often.")
@click.option("--num-envs", default=8, type=click.IntRange(min=1), show_default=True, help="Parallel environments")
@click.option("--deterministic", is_flag=True, default=False, help="Act greedily (distribution mode)")
@click.option("--seed", default=None, type=int, help="Seed for env resets and sampling")
@click.option("--output", "-o", default=None, type=click.Path(dir_okay=False),
              help="Write the machine-readable result as JSON")
def eval_cmd(
    config: str,
    agents: tuple[str, ...],
    layouts: tuple[str, ...],
    num_matches: int,
    num_envs: int,
    deterministic: bool,
    seed: int | None,
    output: str | None,
) -> None:
    """Evaluate agents/checkpoints against each other (no training).

    Exit code: 0 done, 1 config error (also a malformed checkpoint, a role signature or weights
    that do not fit), 2 bad command-line arguments, 130 SIGINT (Ctrl+C), 143 SIGTERM.
    """
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if output is not None and not Path(output).parent.is_dir():
        raise click.BadParameter(f"directory {str(Path(output).parent)!r} does not exist", param_hint="'--output'")
    with _config_errors():
        from colosseum.sp2.core.config import load_config
        from colosseum.sp2.core.registry import env_spec, validate_config
        from colosseum.sp2.eval import evaluate

        cfg = load_config(config)
        validate_config(cfg)
        specs = [_parse_agent_spec(spec) for spec in agents]
        names = [name for name, _ in specs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise click.BadParameter(f"duplicate agent names {duplicates}", param_hint="'--agent'")
        for _name, path in specs:
            p = Path(path)
            if not (p.is_dir() or (p.is_file() and p.suffix == ".pt")):
                raise click.BadParameter(f"{p}: expected a checkpoint directory or a .pt file",
                                         param_hint="'--agent'")
        spec = env_spec(cfg)
        chosen = list(layouts) or list(spec.layouts)
        competitive = any(spec.num_teams(name) >= 2 for name in chosen if name in spec.layouts)
        if len(specs) > 1 and competitive and num_matches % 2:
            click.echo(f"Note: --num-matches {num_matches} is odd; using {num_matches + 1} per pair so every "
                       f"agent plays every side equally often.", err=True)
            num_matches += 1
        report = evaluate(cfg, dict(specs), layouts=list(layouts) or None, num_matches=num_matches, seed=seed,
                          deterministic=deterministic, num_envs=num_envs)
    click.echo("\n" + report.text())
    if output:
        report.write_json(output)
        click.echo(f"Result written to {output}")


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--data", "-d", required=True, type=click.Path(exists=True),
              help="BC data: a .pt file or a directory of .pt files (keys: observations, actions, "
                   "optional action_masks, dones; trees in the agent's spaces)")
@click.option("--output", "-o", required=True, type=click.Path(), help="Where to save the trained state_dict (.pt)")
@click.option("--agent", "-a", "agent", default=None,
              help="Agent whose networks and roles are trained (default: the config's only trainable agent)")
@click.option("--epochs", default=10, type=click.IntRange(min=1), show_default=True, help="Number of BC epochs")
@click.option("--batch-size", default=256, type=click.IntRange(min=1), show_default=True,
              help="Decisions per gradient step")
@click.option("--lr", default=1e-3, type=float, show_default=True, help="Adam learning rate")
@click.option("--seq-len", default=None, type=click.IntRange(min=1),
              help="Window length for stateful models (default: bc.seq_len from the config, 64)")
def bc(
    config: str,
    data: str,
    output: str,
    agent: str | None,
    epochs: int,
    batch_size: int,
    lr: float,
    seq_len: int | None,
) -> None:
    """Train one agent's policy by offline behavioral cloning (loss = -log pi(a|s), masks applied).

    Exit code: 0 done, 1 config or data error, 2 bad command-line arguments, 130 SIGINT.
    """
    import logging

    import torch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    with _config_errors():
        from colosseum.core.errors import ConfigError
        from colosseum.sp2.bc.offline_bc import OfflineBCTrainer
        from colosseum.sp2.core.config import load_config
        from colosseum.sp2.core.registry import build_model, env_spec, validate_config
        from colosseum.sp2.core.roles import agent_role_spec, resolve_agent_roles
        from colosseum.sp2.core.specs import ActionSpec, ObsSpec

        cfg = load_config(config)
        validate_config(cfg)
        agent_ids = cfg.get_trainable_agent_ids()
        if agent is None:
            if len(agent_ids) != 1:
                raise ConfigError(f"the config has {len(agent_ids)} trainable agents {agent_ids}; "
                                  f"choose one with --agent")
            agent = agent_ids[0]
        elif agent not in agent_ids:
            raise ConfigError(f"--agent {agent!r} is not a trainable agent of the config ({agent_ids})")
        spec = env_spec(cfg)
        role = agent_role_spec(spec, resolve_agent_roles(cfg, spec)[agent])
        agent_cfg = cfg.get_agent_config(agent)
        model = build_model(agent_cfg, role)
    device = agent_cfg.learner.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = OfflineBCTrainer(
        model, ActionSpec.from_space(role.action_space), ObsSpec.from_space(role.observation_space),
        lr=lr, device=device, seq_len=seq_len if seq_len is not None else agent_cfg.bc.seq_len,
    )
    with _config_errors():  # unreadable or malformed data, actions that do not fit the policy (DataError)
        trainer.load_data(data)
        metrics = trainer.train(num_epochs=epochs, batch_size=batch_size)

    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output)
    message = f"BC training complete ({agent}): final-epoch NLL={metrics['bc_loss']:.4f}"
    if "accuracy" in metrics:
        message += f", accuracy={metrics['accuracy']:.3f}"
    click.echo(message)
    click.echo(f"Weights saved to {output}")


if __name__ == "__main__":
    main()
