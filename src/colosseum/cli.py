"""CLI entry point (the ``colosseum`` console script and ``python -m colosseum``)."""

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
    turn it into "Aborted!" with exit code 1). Wraps every command (``_InterruptibleGroup``).

    A Ctrl-C that Python could not raise here (it landed in a weakref callback or similar) is
    recorded instead of printed: the run's signal handling takes it over (``train``), and a
    command that finishes anyway still ends with "Interrupted" and 130.
    """
    try:
        from colosseum.utils.process import catch_lost_interrupts, take_lost_interrupt

        with catch_lost_interrupts():
            marker = os.environ.get(_STARTUP_MARKER_ENV)
            if marker:
                open(marker, "a").close()
            yield
        if take_lost_interrupt():
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        _forget_unhandled_keyboard_interrupt()
        click.echo("Interrupted", err=True)
        sys.exit(130)


def _forget_unhandled_keyboard_interrupt() -> None:
    """Clear CPython's "KeyboardInterrupt was unhandled" flag for an interrupt we did handle.

    A KeyboardInterrupt that leaves a string ``exec``/``eval`` (``PyRun_String*``) sets that
    flag even when it is caught further up, and under ``python -m`` the interpreter then ends
    the process with SIGINT (exit -2) instead of our ``sys.exit(130)``. ``import torch`` builds
    hundreds of dataclass methods with ``exec(str)``, so a Ctrl-C during startup often lands in
    one (FIX-2). Every string ``exec`` resets the flag when it starts, so a trivial one that
    completes clears it.
    """
    exec("pass", {})


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
    """A config (or env contract, or input data) problem found at startup, or a scripted player's error:
    one line on stderr, exit code 1, no traceback."""
    from colosseum.core.errors import ConfigError, EnvContractError, PlayerError

    try:
        yield
    except (ConfigError, EnvContractError) as e:
        click.echo(f"Config error: {e}", err=True)
        sys.exit(1)
    except PlayerError as e:
        click.echo(f"Player error: {e}", err=True)
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


def _parse_agent_spec(spec: str) -> tuple[str, str | None]:
    """``name=path`` -> (name, path); ``name`` -> (name, None): a scripted or frozen agent of the config."""
    name, sep, path = spec.partition("=")
    if not name or (sep and not path):
        raise click.BadParameter(f"expected name or name=path, got {spec!r}", param_hint="'--agent'")
    return name, (path if sep else None)


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--set", "overrides", multiple=True,
              help="Override config values (e.g., --set rollout.num_workers=8)." + _SET_HELP_YAML)
def train(config: str, overrides: tuple[str, ...]) -> None:
    """Train the config's agents (single machine).

    Exit code: 0 budget reached, 1 config error or a child process died, 130 SIGINT, 143 SIGTERM.
    """
    with _config_errors():
        from colosseum.launcher import run_training  # imports torch: about a second

        code = run_training(config, overrides=_parse_overrides(overrides) or None)
    sys.exit(code)


@main.command("validate")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--set", "overrides", multiple=True,
              help="Override config values (e.g., --set env.max_idle_steps=200)." + _SET_HELP_YAML)
def validate_cmd(config: str, overrides: tuple[str, ...]) -> None:
    """Validate a config: GameSpec, roles, matchmaking, env steps under the contract, every agent's model,
    every scripted agent played under the legality gate, every frozen agent loaded."""
    with _config_errors():
        from colosseum.core.config import load_config
        from colosseum.core.registry import validate_config

        cfg = load_config(config, _parse_overrides(overrides) or None)
        report = validate_config(cfg)
        for aid in cfg.agent_ids():
            click.echo(f"  OK: agent '{aid}' ({cfg.agent_kind(aid)})")
        for line in report.lines:
            click.echo(f"  {line}")
    click.echo("Config is valid.")


@main.command("eval")
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Config YAML: its env is used for every match; agents.<name> / networks build .pt agents")
@click.option("--agent", "-a", "agents", required=True, multiple=True,
              help="name=path or name. path is a checkpoint dir (architecture and roles from its meta.json) or "
                   "a .pt state_dict (architecture and roles of agents.<name>, else the global networks playing "
                   "every role). A bare name is a scripted or frozen agent of the config. Repeatable.")
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
        from colosseum.core.config import load_config
        from colosseum.core.registry import env_spec, validate_config
        from colosseum.eval import evaluate

        cfg = load_config(config)
        validate_config(cfg)
        specs = [_parse_agent_spec(spec) for spec in agents]
        names = [name for name, _ in specs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise click.BadParameter(f"duplicate agent names {duplicates}", param_hint="'--agent'")
        for name, path in specs:
            if path is None:
                if name not in cfg.agent_ids():
                    raise click.BadParameter(f"{name!r} is not an agent of the config; use name=path for a "
                                             f"checkpoint dir or a .pt file", param_hint="'--agent'")
                continue
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


@main.command("record")
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Config YAML: its env is used for every match; its scripted/frozen agents can be named")
@click.option("--player", "-p", required=True,
              help="Who is recorded: a scripted or frozen agent of the config, or name=path (checkpoint dir or .pt)")
@click.option("--against", "against", multiple=True,
              help="Opponent (same forms as --player; repeatable): each one plays the player in eval's pair "
                   "rotation and only the player's seats are recorded. Without --against the player takes every "
                   "seat and every seat is recorded.")
@click.option("--layout", "layouts", multiple=True,
              help="Layout to play (repeatable). Default: every layout the players can fill.")
@click.option("--num-matches", "-n", default=100, type=click.IntRange(min=1), show_default=True,
              help="Matches per layout (with --against: per opponent and layout). With --against and a layout "
                   "of two or more teams an odd count is rounded up, so the player plays every side equally "
                   "often.")
@click.option("--output", "-o", required=True, type=click.Path(file_okay=False),
              help="New or empty directory: <output>/<role>/part-NNNNN.pt (BC data) and record.json")
@click.option("--num-envs", default=8, type=click.IntRange(min=1), show_default=True, help="Parallel environments")
@click.option("--seed", default=None, type=int, help="Seed for env resets, bots and sampling")
@click.option("--deterministic", is_flag=True, default=False,
              help="Neural players act greedily (scripted bots are unaffected)")
def record_cmd(config: str, player: str, against: tuple[str, ...], layouts: tuple[str, ...], num_matches: int,
               output: str, num_envs: int, seed: int | None, deterministic: bool) -> None:
    """Record a player's decisions as behavioural-cloning data (read by 'colosseum bc --data <output>').

    Exit code: 0 done, 1 config error (unknown player, a layout the player cannot fill without
    --against, a non-empty output directory) or a scripted player's illegal action, 2 bad
    command-line arguments, 130 SIGINT, 143 SIGTERM.
    """
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    with _config_errors():  # also a scripted player's error (PlayerError)
        from colosseum.core.config import load_config
        from colosseum.core.registry import validate_config
        from colosseum.eval import EvalReport
        from colosseum.record import RECORD_FILE, record

        cfg = load_config(config)
        validate_config(cfg)
        content = record(cfg, player, list(against), layouts=list(layouts) or None, num_matches=num_matches,
                         output=output, num_envs=num_envs, seed=seed, deterministic=deterministic)
    summary = content["summary"]
    report = EvalReport(agents=summary["agents"], num_matches=summary["num_matches"],
                        deterministic=summary["deterministic"], layouts=summary["layouts"])
    click.echo("\n" + report.text())
    click.echo(f"Recorded {content['decisions']} decisions ({content['seat_episodes']} seat-episodes of "
               f"{content['matches']} matches) into {output}:")
    for role, cell in content["roles"].items():
        click.echo(f"  {role}: {cell['seat_episodes']} seat-episodes, {cell['decisions']} decisions, "
                   f"{len(cell['files'])} file(s)")
    click.echo(f"Description: {output}/{RECORD_FILE}")


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--data", "-d", "data", required=True, multiple=True, type=click.Path(exists=True),
              help="BC data, repeatable: a .pt file, a directory of .pt files (keys: observations, actions, "
                   "optional action_masks, dones; trees in the agent's spaces), or a 'colosseum record' output "
                   "directory (its record.json selects the folders of the agent's roles)")
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
    data: tuple[str, ...],
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
        from colosseum.bc.offline_bc import OfflineBCTrainer
        from colosseum.core.config import load_config
        from colosseum.core.errors import ConfigError
        from colosseum.core.registry import build_model, env_spec, validate_config
        from colosseum.core.roles import agent_role_spec, resolve_agent_roles
        from colosseum.core.specs import ActionSpec, ObsSpec

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
        roles = resolve_agent_roles(cfg, spec)[agent]
        role = agent_role_spec(spec, roles)
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
        from colosseum.bc.offline_bc import bc_data_sources

        for source in bc_data_sources(data, roles):
            trainer.load_data(source)
        metrics = trainer.train(num_epochs=epochs, batch_size=batch_size)

    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output)
    message = (f"BC training complete ({agent}): {int(metrics['num_samples'])} decisions, "
               f"final-epoch NLL={metrics['bc_loss']:.4f}")
    if "accuracy" in metrics:
        message += f", accuracy={metrics['accuracy']:.3f}"
    click.echo(message)
    click.echo(f"Weights saved to {output}")


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
@click.option("--set", "overrides", multiple=True, help="Override config values." + _SET_HELP_YAML)
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
    """Start a gRPC weight store server. Ctrl-C stops it with exit code 130."""
    import logging

    from colosseum.weight_store.grpc_store import serve_weight_store

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    server = serve_weight_store(port=port, max_message_mb=max_message_mb)
    click.echo(f"Weight store serving on port {port}. Press Ctrl+C to stop.")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)
        raise  # "Interrupted" and exit code 130, like every command (_interrupts)


if __name__ == "__main__":
    main()
