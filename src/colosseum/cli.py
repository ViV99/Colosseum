"""CLI entry point for Colosseum."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import click

# Test hook: when set, the CLI touches this file when ``_interrupts`` becomes active, i.e.
# from then on a Ctrl-C is handled by it (lifecycle tests send SIGINT during startup only
# after the file exists).
_STARTUP_MARKER_ENV = "COLOSSEUM_TEST_STARTUP_MARKER"


@contextmanager
def _interrupts() -> Iterator[None]:
    """Ctrl-C before the run installs its own signal handling (imports, config loading,
    validation): one line on stderr and exit code 130, as after a handled SIGINT (click
    would turn it into "Aborted!" with exit code 1). Wraps every command (see
    ``_InterruptibleGroup``), so each command's imports and work are inside it."""
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
    # Spawned children inherit sys.path (R5-19).
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


@contextmanager
def _config_errors() -> Iterator[None]:
    """A config (or env contract, or input data) problem found at startup: one line on stderr,
    exit code 1, no traceback (D10)."""
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
    from colosseum.core.config import parse_override_value

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
    """Train an agent using the specified configuration.

    Exit code: 0 budget reached, 1 config error or a child process died, 130 SIGINT, 143 SIGTERM.
    """
    with _config_errors():
        from colosseum.launcher import run_training  # imports torch: about a second

        code = run_training(config, overrides=_parse_overrides(overrides) or None)
    sys.exit(code)


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--data", "-d", required=True, type=click.Path(exists=True),
              help="BC data: a .pt file or a directory of .pt files (keys: observations, actions, "
                   "optional action_masks, dones)")
@click.option("--output", "-o", required=True, type=click.Path(), help="Where to save the trained state_dict (.pt)")
@click.option("--epochs", default=10, type=int, show_default=True, help="Number of BC epochs")
@click.option("--batch-size", default=256, type=int, show_default=True, help="Transitions per gradient step")
@click.option("--lr", default=1e-3, type=float, show_default=True, help="Adam learning rate")
@click.option("--seq-len", default=None, type=click.IntRange(min=1),
              help="Window length for stateful models (default: bc.seq_len from the config, 64)")
def bc(
    config: str,
    data: str,
    output: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seq_len: int | None,
) -> None:
    """Train a policy by offline behavioral cloning (loss = -log pi(a|s), masks applied)."""
    import logging

    import torch

    from colosseum.bc.offline_bc import OfflineBCTrainer
    from colosseum.core.config import load_config
    from colosseum.core.registry import build_model, validate_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    with _config_errors():
        cfg = load_config(config)
        validate_config(cfg)
    model = build_model(cfg)
    device = cfg.learner.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = OfflineBCTrainer(
        model, lr=lr, device=device,
        seq_len=seq_len if seq_len is not None else cfg.bc.seq_len,
    )
    with _config_errors():  # unreadable or malformed data, actions that do not fit the policy (DataError)
        trainer.load_data(data)
        metrics = trainer.train(num_epochs=epochs, batch_size=batch_size)

    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output)
    message = f"BC training complete: final-epoch NLL={metrics['bc_loss']:.4f}"
    if "accuracy" in metrics:
        message += f", accuracy={metrics['accuracy']:.3f}"
    click.echo(message)
    click.echo(f"Weights saved to {output}")


@main.command("eval")
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Config YAML: its env is used for every match; its networks (always validated) build "
                   ".pt agents and checkpoint dirs without meta.json 'networks'")
@click.option("--agent", "-a", "agents", required=True, multiple=True,
              help="name=path. path is a checkpoint dir (model built from its meta.json 'networks', "
                   "else from --config) or a .pt state_dict (model built from --config). Repeatable.")
@click.option("--num-matches", "-n", default=100, type=click.IntRange(min=1), show_default=True,
              help="Matches per agent pair (solo: episodes per agent). An odd pairwise count is rounded "
                   "up to the next even number, so every agent plays every seat equally often.")
@click.option("--num-envs", default=8, type=click.IntRange(min=1), show_default=True, help="Parallel environments")
@click.option("--deterministic", is_flag=True, default=False, help="Act greedily (distribution mode)")
@click.option("--seed", default=None, type=int, help="Seed for env resets and sampling")
@click.option("--output", "-o", default=None, type=click.Path(dir_okay=False),
              help="Write the machine-readable result as JSON")
def eval_cmd(
    config: str,
    agents: tuple[str, ...],
    num_matches: int,
    num_envs: int,
    deterministic: bool,
    seed: int | None,
    output: str | None,
) -> None:
    """Evaluate agents/checkpoints against each other (no training).

    Exit code: 0 done, 1 config error (also a malformed checkpoint or mismatched weights),
    2 bad command-line arguments, 130 SIGINT (Ctrl+C), 143 SIGTERM.
    """
    import logging
    from pathlib import Path

    from colosseum.core.config import load_config
    from colosseum.core.registry import import_class, validate_config
    from colosseum.eval import evaluate, load_eval_model

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if output is not None and not Path(output).parent.is_dir():
        raise click.BadParameter(f"directory {str(Path(output).parent)!r} does not exist", param_hint="'--output'")

    with _config_errors():
        cfg = load_config(config)
        validate_config(cfg)
        specs = [_parse_agent_spec(spec) for spec in agents]
        names = [name for name, _ in specs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise click.BadParameter(f"duplicate agent names {duplicates}", param_hint="'--agent'")
        models = {}
        validated = {cfg.networks.model_dump_json()}  # validate_config(cfg) above
        for name, path in specs:
            try:
                models[name] = load_eval_model(path, cfg, validated=validated)
            except FileNotFoundError as exc:
                raise click.BadParameter(str(exc), param_hint="'--agent'") from exc

        if len(models) > 1 and cfg.env.num_players > 1 and num_matches % 2:
            click.echo(f"Note: --num-matches {num_matches} is odd; using {num_matches + 1} per pair "
                       f"so every agent plays every seat equally often.", err=True)
            num_matches += 1

        env_cls = import_class(cfg.env.env_class)

        def env_fn():
            return env_cls(**cfg.env.kwargs)

        report = evaluate(models, env_fn, num_matches=num_matches, num_envs=num_envs,
                          deterministic=deterministic, seed=seed)
    click.echo("\n" + report.summary())
    if output:
        report.write_json(output)
        click.echo(f"Result written to {output}")


@main.command("validate")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--set", "overrides", multiple=True,
              help="Override config values (e.g., --set env.num_players=2)." + _SET_HELP_YAML)
def validate_cmd(config: str, overrides: tuple[str, ...]) -> None:
    """Validate a config: schema, env num_players, and a dummy forward of every agent's model."""
    from colosseum.core.config import load_config
    from colosseum.core.registry import validate_config

    with _config_errors():
        cfg = load_config(config, _parse_overrides(overrides) or None)
        for aid in cfg.get_trainable_agent_ids():
            validate_config(cfg.get_agent_config(aid))
            click.echo(f"  OK: agent '{aid}'")
    click.echo("Config is valid.")


@main.command("run-learner")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--agent", "-a", default="agent_0", help="Trainable agent id this learner owns")
@click.option("--traj-port", default=50052, type=int, help="Port for this learner's TrajectoryService")
@click.option("--weight-store", required=True, help="WeightStore address host:port")
@click.option("--set", "overrides", multiple=True,
              help="Override config values (e.g., --set rollout.num_workers=8)." + _SET_HELP_YAML)
def run_learner_cmd(config: str, agent: str, traj_port: int, weight_store: str, overrides: tuple[str, ...]) -> None:
    """Run one agent's learner as a gRPC service (distributed mode)."""
    with _config_errors():
        from colosseum.distributed import run_distributed_learner

        code = run_distributed_learner(config, agent, traj_port, weight_store,
                                       overrides=_parse_overrides(overrides) or None)
    sys.exit(code)


@main.command("run-workers")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--weight-store", required=True, help="WeightStore address host:port")
@click.option("--learner", "-l", "learners", required=True, multiple=True,
              help="Learner address per agent: agent_id=host:port (repeatable)")
@click.option("--set", "overrides", multiple=True,
              help="Override config values." + _SET_HELP_YAML)
def run_workers_cmd(config: str, weight_store: str, learners: tuple[str, ...], overrides: tuple[str, ...]) -> None:
    """Run rollout workers feeding remote learners over gRPC (distributed mode)."""
    learner_addresses: dict[str, str] = {}
    for spec in learners:
        if "=" not in spec:
            raise click.BadParameter(f"--learner must be agent_id=host:port, got {spec!r}")
        aid, addr = spec.split("=", 1)
        learner_addresses[aid] = addr

    with _config_errors():
        from colosseum.distributed import run_distributed_workers

        code = run_distributed_workers(config, weight_store, learner_addresses,
                                       overrides=_parse_overrides(overrides) or None)
    sys.exit(code)


@main.command("serve-weight-store")
@click.option("--port", default=50051, type=int, help="gRPC port")
@click.option("--max-message-mb", default=64, type=int, help="Max gRPC message size in MiB")
def serve_weight_store_cmd(port: int, max_message_mb: int) -> None:
    """Start a gRPC weight store server."""
    import logging

    from colosseum.weight_store.grpc_store import serve_weight_store

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    server = serve_weight_store(port=port, max_message_mb=max_message_mb)
    click.echo(f"Weight store serving on port {port}. Press Ctrl+C to stop.")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    main()
