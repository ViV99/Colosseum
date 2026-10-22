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


_RESUME_HELP = ("Continue the run dir RUN in place (same dir and run id; refused while it runs). -c is optional: "
                "without it the run's saved config is used; --set applies either way.")


@main.command()
@click.option("--config", "-c", default=None, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--resume", "resume", default=None, metavar="RUN", type=click.Path(exists=True, file_okay=False),
              help=_RESUME_HELP)
@click.option("--listen", "listen", is_flag=False, flag_value="0.0.0.0", default=None, metavar="[ADDR]",
              help="Also accept other machines' nodes on ADDR:distributed.port (TLS); without ADDR all interfaces")
@click.option("--set", "overrides", multiple=True,
              help="Override config values (e.g., --set rollout.num_workers=8)." + _SET_HELP_YAML)
def train(config: str | None, resume: str | None, listen: str | None, overrides: tuple[str, ...]) -> None:
    """Train the config's agents: a hub and a local node on this machine (other machines join with --listen).

    Exit code: 0 budget reached, 1 config error or a failure (a child error, a dead process), 130 SIGINT, 143 SIGTERM.
    """
    if config is None and resume is None:
        raise click.UsageError("Missing option '-c' / '--config'.")
    from colosseum.utils.process import limit_main_blas_threads

    limit_main_blas_threads()   # first: numpy is imported below; this process's BLAS gets one thread (SP4)
    with _config_errors():
        from colosseum.hub.hub import run_training  # imports torch: about a second

        code = run_training(config, overrides=_parse_overrides(overrides) or None, listen=listen, resume=resume)
    sys.exit(code)


@main.group("hub", invoke_without_command=True)
@click.option("--config", "-c", default=None, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--listen", "listen", default=None, metavar="ADDR",
              help="Listen only on ADDR (default: all interfaces)")
@click.option("--port", default=None, type=click.IntRange(0, 65535),
              help="The hub's port (default distributed.port; 0: an ephemeral port, see runs/<name>/hub.json)")
@click.option("--insecure", is_flag=True, default=False, help="No TLS (trusted LAN only); the token stays required")
@click.option("--resume", "resume", default=None, metavar="RUN", type=click.Path(exists=True, file_okay=False),
              help=_RESUME_HELP + " Without --port it listens on the port of the run's previous hub.json.")
@click.option("--set", "overrides", multiple=True, help="Override config values." + _SET_HELP_YAML)
@click.pass_context
def hub_group(ctx: click.Context, config: str | None, listen: str | None, port: int | None, insecure: bool,
              resume: str | None, overrides: tuple[str, ...]) -> None:
    """Run a hub that machines join (colosseum join, T6.3); exit 0 at the budget, 1 on errors, 130 / 143 on signals."""
    if ctx.invoked_subcommand is not None:
        return
    if config is None and resume is None:
        raise click.UsageError("colosseum hub needs --config / -c (or --resume RUN)")
    from colosseum.utils.process import limit_main_blas_threads

    limit_main_blas_threads()   # first: numpy is imported below (as for train)
    with _config_errors():
        from colosseum.hub.hub import run_hub  # imports torch: about a second

        code = run_hub(config, overrides=_parse_overrides(overrides) or None, listen=listen, port=port,
                       insecure=insecure, resume=resume)
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
              help="Config YAML: its env is used for every match; a trainable agents.<name> or the global "
                   "networks build .pt agents")
@click.option("--agent", "-a", "agents", required=True, multiple=True,
              help="name=path or name. path is a checkpoint dir (architecture and roles from its meta.json) or "
                   "a .pt state_dict (architecture and roles of the trainable agents.<name>, else the global "
                   "networks playing every role; a frozen entry of that name is not used for a path given here). "
                   "A bare name is a scripted or frozen agent of the config. Repeatable.")
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
@click.option("--set", "overrides", multiple=True,
              help="Override config values (e.g., --set env.max_idle_steps=200)." + _SET_HELP_YAML)
def eval_cmd(
    config: str,
    agents: tuple[str, ...],
    layouts: tuple[str, ...],
    num_matches: int,
    num_envs: int,
    deterministic: bool,
    seed: int | None,
    output: str | None,
    overrides: tuple[str, ...],
) -> None:
    """Evaluate agents/checkpoints against each other (no training).

    The config is checked for what eval uses: the env, the trainable agents' models and the scripted /
    frozen agents named with -a (warm-start sources and other fixed agents are not read).

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
        from colosseum.core.errors import ConfigError
        from colosseum.core.registry import env_spec, validate_config
        from colosseum.eval import balanced_match_count, evaluate, parse_agent_spec

        cfg = load_config(config, _parse_overrides(overrides) or None)
        try:
            specs = [parse_agent_spec(spec, option=None) for spec in agents]   # click names --agent
        except ConfigError as e:
            raise click.BadParameter(str(e), param_hint="'--agent'") from e
        report_v = validate_config(cfg, scope="play", players=[name for name, path in specs if path is None])
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
        if len(specs) > 1:
            rounded = balanced_match_count(spec, list(layouts) or None, num_matches)
            if rounded != num_matches:
                click.echo(f"Note: --num-matches {num_matches} is odd; using {rounded} per pair so every "
                           f"agent plays every side equally often.", err=True)
                num_matches = rounded
        report = evaluate(cfg, dict(specs), layouts=list(layouts) or None, num_matches=num_matches, seed=seed,
                          deterministic=deterministic, num_envs=num_envs, fixed=report_v.fixed)
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
@click.option("--set", "overrides", multiple=True,
              help="Override config values (e.g., --set env.max_idle_steps=200)." + _SET_HELP_YAML)
def record_cmd(config: str, player: str, against: tuple[str, ...], layouts: tuple[str, ...], num_matches: int,
               output: str, num_envs: int, seed: int | None, deterministic: bool, overrides: tuple[str, ...]) -> None:
    """Record a player's decisions as behavioural-cloning data (read by 'colosseum bc --data <output>').

    The config is checked for what record uses: the env, the trainable agents' models and the scripted /
    frozen agents named by --player / --against (warm-start sources and other fixed agents are not read).

    Exit code: 0 done, 1 config error (unknown player, a layout the player cannot fill without
    --against, a non-empty output directory) or a scripted player's illegal action, 2 bad
    command-line arguments, 130 SIGINT, 143 SIGTERM.
    """
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    with _config_errors():  # also a scripted player's error (PlayerError)
        from colosseum.core.config import load_config
        from colosseum.core.registry import validate_config
        from colosseum.eval import EvalReport, parse_agent_spec
        from colosseum.record import RECORD_FILE, record

        cfg = load_config(config, _parse_overrides(overrides) or None)
        named = [parse_agent_spec(text, option=opt) for opt, text in (("--player", player),
                                                                       *(("--against", a) for a in against))]
        report_v = validate_config(cfg, scope="play", players=[name for name, path in named if path is None])
        content = record(cfg, player, list(against), layouts=list(layouts) or None, num_matches=num_matches,
                         output=output, num_envs=num_envs, seed=seed, deterministic=deterministic,
                         fixed=report_v.fixed)
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


class _TournamentCommand(click.Command):
    """Keeps the raw arguments: every ``--agent`` / ``--snapshots`` belongs to the ``--run`` before it."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        ctx.meta["tournament_args"] = list(args)
        return super().parse_args(ctx, args)


def _group_runs(args: list[str]) -> list[dict]:
    """``[{"run", "agent", "snapshots"}]`` in command-line order (``--opt value`` and ``--opt=value``)."""
    runs: list[dict] = []
    i = 0
    while i < len(args):
        token, key, value = args[i], None, None
        for opt in ("--run", "--agent", "--snapshots"):
            if token == opt:
                key, value = opt, args[i + 1] if i + 1 < len(args) else None
                i += 1
                break
            if token.startswith(opt + "="):
                key, value = opt, token[len(opt) + 1:]
                break
        i += 1
        if key is None:
            continue
        if key == "--run":
            runs.append({"run": value, "agent": None, "snapshots": None})
            continue
        if not runs:
            raise click.UsageError(f"{key} belongs to a --run: write it after the --run it describes")
        slot = key.removeprefix("--")
        if runs[-1][slot] is not None:
            raise click.UsageError(f"--run {runs[-1]['run']}: {key} given twice")
        runs[-1][slot] = value
    return runs


def _run_selection(item: dict):
    from pathlib import Path

    from colosseum.arena.tournament.players import RunSelection

    text = item["run"]
    alias, sep, path = text.partition("=")
    if not sep or not alias or "/" in alias or not path:
        alias, path = None, text
    p = Path(path)
    if not p.is_dir():
        raise click.BadParameter(f"{path}: not a directory", param_hint="'--run'")
    return RunSelection(path=p, alias=alias, agent=item["agent"], snapshots=item["snapshots"] or "all")


@main.command("tournament", cls=_TournamentCommand)
@click.option("--config", "-c", default=None, type=click.Path(exists=True, dir_okay=False),
              help="Config YAML (its env and agents); default: the first --run's config.resolved.yaml")
@click.option("-a", "--candidate", "candidates", multiple=True,
              help="Candidate: name=path (a checkpoint dir of any architecture, or a .pt built like eval's: the "
                   "trainable agents.<name>, else the global networks) or a scripted / frozen agent of the config. "
                   "Repeatable.")
@click.option("--run", "runs", multiple=True,
              help="[alias=]runs/x: candidates are its stored snapshots (repeatable; --agent / --snapshots after a "
                   "--run apply to it). Matches go to the first run's matches.db, reports to its tournaments/<name>/.")
@click.option("--agent", "run_agents", multiple=True, help="The --run's agent (default: its only trainable agent)")
@click.option("--snapshots", "run_snapshots", multiple=True,
              help="The --run's snapshots: all (default), last:N, top:N (by the run journal's rating lower bound), "
                   "every:K")
@click.option("--field", default=None,
              help="Comma-separated opponents: agents of the config, -a names or snapshots [alias/]agent@vN. "
                   "Default: every scripted / frozen agent of the config.")
@click.option("--mode", default="gauntlet", show_default=True,
              type=click.Choice(["gauntlet", "round-robin", "halving", "select"]))
@click.option("--layout", "layouts", multiple=True, help="Layout (repeatable); default: the enabled layouts")
@click.option("--blocks", default=None, type=click.IntRange(min=1),
              help="gauntlet / round-robin: blocks per composition (default 50; T8.2 ruling)")
@click.option("--halving-blocks", default=None, type=click.IntRange(min=1),
              help="halving / select: blocks per composition of round 0 (default 8; T8.2 ruling)")
@click.option("--confirm-blocks", default=None, type=click.IntRange(min=1),
              help="select: blocks per composition of the confirmation stage (default 100; T8.2 ruling)")
@click.option("--finalists", default=None, type=click.IntRange(min=1), help="halving / select (default 3)")
@click.option("--num-workers", default=1, type=click.IntRange(min=1), show_default=True, help="Worker processes")
@click.option("--num-envs", default=8, type=click.IntRange(min=1), show_default=True, help="Envs per worker")
@click.option("--deterministic", is_flag=True, default=False, help="Greedy policies (default: stochastic)")
@click.option("--seed", default=0, type=int, show_default=True, help="Seed of the block seeds (domains A, B)")
@click.option("--name", default=None, help="Tournament name (default <mode>-<YYYYmmdd-HHMMSS>); the same name "
                                           "continues an interrupted tournament, with the same settings (every "
                                           "size flag counts, also one the mode does not use)")
@click.option("--output", "-o", default=None, type=click.Path(file_okay=False),
              help="Without --run: the directory for matches.db and tournaments/<name>/")
@click.option("--pin", is_flag=True, default=False, help="select with --run: pin the confirmed winner snapshot")
@click.option("--set", "overrides", multiple=True, help="Override config values." + _SET_HELP_YAML)
def tournament_cmd(config: str | None, candidates: tuple[str, ...], runs: tuple[str, ...], run_agents: tuple[str, ...],
                   run_snapshots: tuple[str, ...], field: str | None, mode: str, layouts: tuple[str, ...],
                   blocks: int | None, halving_blocks: int | None, confirm_blocks: int | None, finalists: int | None,
                   num_workers: int, num_envs: int, deterministic: bool, seed: int, name: str | None,
                   output: str | None, pin: bool, overrides: tuple[str, ...]) -> None:
    """Compare candidates (run snapshots, checkpoint dirs of any architecture, .pt weights, scripted / frozen
    agents) on common seeds with seat rotations, and select the best (--mode select: screening, then confirmation
    on fresh seeds).

    The config is checked in the "play" scope (like eval). Exit code: 0 done (also a select whose winner is not
    confirmed), 1 config, player or env error, a dead worker or a --pin winner the run evicted meanwhile (the
    reports are written first), 2 bad arguments, 130 SIGINT (finished blocks are kept: rerun with the same --name
    to continue).
    """
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    grouped = _group_runs(click.get_current_context().meta.get("tournament_args", []))
    if len(grouped) != len(runs):
        raise click.UsageError("could not match --agent / --snapshots to their --run")
    if pin and (not runs or mode != "select"):
        raise click.UsageError("--pin needs --mode select and --run (it pins the winner in the run's storage)")
    if output is not None and runs:
        raise click.UsageError("--output is for tournaments without --run (with --run the reports go to "
                               "runs/x/tournaments/<name>/)")
    if not runs and config is None:
        raise click.UsageError("give -c CONFIG or --run DIR")
    if not runs and output is None:
        raise click.UsageError("--output DIR is required without --run (it gets matches.db and "
                               "tournaments/<name>/)")
    with _config_errors():
        from colosseum.arena.tournament import modes
        from colosseum.arena.tournament.engine import TournamentError
        from colosseum.arena.tournament.run import TournamentOptions, run_tournament
        from colosseum.core.errors import ConfigError
        from colosseum.eval import parse_agent_spec

        try:
            parsed = tuple(parse_agent_spec(text, option=None) for text in candidates)   # click names -a
        except ConfigError as e:
            raise click.BadParameter(str(e), param_hint="'-a'") from e
        options = TournamentOptions(
            mode=mode, candidates=parsed, runs=tuple(_run_selection(item) for item in grouped),
            field=None if field is None else tuple(n.strip() for n in field.split(",") if n.strip()),
            layouts=tuple(layouts),
            blocks=blocks or modes.DEFAULT_BLOCKS, halving_blocks=halving_blocks or modes.DEFAULT_HALVING_BLOCKS,
            confirm_blocks=confirm_blocks or modes.DEFAULT_CONFIRM_BLOCKS,
            finalists=finalists or modes.DEFAULT_FINALISTS, seed=seed, deterministic=deterministic,
            num_workers=num_workers, num_envs=num_envs, name=name, output=output, pin=pin, config_path=config,
            overrides=_parse_overrides(overrides))
        try:
            outcome = run_tournament(options)
        except TournamentError as e:
            click.echo(f"Tournament error: {e}", err=True)
            sys.exit(1)
    click.echo("\n" + outcome.report.text())
    click.echo(f"Blocks: {outcome.played_blocks} played, {outcome.reused_blocks} from the journal")
    for path in outcome.written:
        click.echo(f"Written: {path}")
    for winner in outcome.pinned:
        click.echo(f"Pinned {winner}")
    for error in outcome.pin_errors:
        click.echo(f"Pin error: {error}", err=True)
    if outcome.pin_errors:
        sys.exit(1)


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
@click.option("--seed", default=None, type=int,
              help="Seed for the model's initial weights and the minibatch order (torch and numpy)")
@click.option("--set", "overrides", multiple=True,
              help="Override config values (e.g., --set bc.seq_len=32)." + _SET_HELP_YAML)
def bc(
    config: str,
    data: tuple[str, ...],
    output: str,
    agent: str | None,
    epochs: int,
    batch_size: int,
    lr: float,
    seq_len: int | None,
    seed: int | None,
    overrides: tuple[str, ...],
) -> None:
    """Train one agent's policy by offline behavioral cloning (loss = -log pi(a|s), masks applied).

    The config is checked for what bc uses: the env and the trainable agents' models (warm-start sources and
    scripted / frozen agents are not read), so the config may already name the output (init.from, a frozen
    agent with path: <output>).

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

        cfg = load_config(config, _parse_overrides(overrides) or None)
        validate_config(cfg, scope="play")
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
        if seed is not None:              # initial weights, window offsets and minibatch order
            import numpy as np

            torch.manual_seed(seed)
            np.random.seed(seed)
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


@main.command("report")
@click.argument("run", type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", default=None, type=click.Path(dir_okay=False),
              help="HTML file to write (default: RUN/report.html)")
def report_cmd(run: str, output: str | None) -> None:
    """Build a self-contained HTML report of a run directory (or of a tournament directory): ratings, results
    matrix, anchors, opponents, training, system, matches and tournaments. The page needs no network.

    Exit code: 0 done, 1 not a run or tournament directory (or an unreadable run), or a section failed (the
    page is still written, with a note in place of each failed section), 2 bad command-line arguments,
    130 SIGINT.
    """
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if output is not None and not Path(output).absolute().parent.is_dir():
        raise click.BadParameter(f"directory {str(Path(output).absolute().parent)!r} does not exist",
                                 param_hint="'--output'")
    with _config_errors():
        from colosseum.arena.report.render import ReportSectionsFailed, write_report

        try:
            path = write_report(run, output)
        except ReportSectionsFailed as e:
            failed = "; ".join(f"{name} ({error})" for name, error in e.failed.items())
            click.echo(f"Report written to {e.path}, but {len(e.failed)} section(s) failed and show a note "
                       f"instead: {failed} (tracebacks in the log above)", err=True)
            sys.exit(1)
    click.echo(f"Report written to {path}")


@main.command("dashboard")
@click.argument("run", type=click.Path(exists=True, file_okay=False))
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Address to listen on. No authentication: keep 127.0.0.1 and use an SSH tunnel from other machines")
@click.option("--port", default=8765, type=click.IntRange(0, 65535), show_default=True,
              help="Port (0: any free port, printed at start)")
@click.option("--refresh", default=10, type=click.IntRange(min=2), show_default=True,
              help="Seconds between page refreshes (at least 2)")
def dashboard_cmd(run: str, host: str, port: int, refresh: int) -> None:
    """Serve the run's report live: the same page as 'colosseum report', refreshed from /api/<section> JSON.
    The match log is opened read-only; ratings come from ratings.json while the run trains.

    Exit code: 1 not a run directory or the port is taken, 2 bad command-line arguments, 130 Ctrl+C.
    """
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    with _config_errors():
        from colosseum.arena.report.server import make_server

        server = make_server(run, host=host, port=port, refresh_sec=refresh)
    click.echo(f"Dashboard of {run} at http://{host}:{server.port}/ (Ctrl+C to stop)")
    try:
        server.serve_forever()
    finally:
        server.server_close()        # a KeyboardInterrupt propagates: "Interrupted", exit code 130


@main.command("pin")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("snapshot", required=False)
@click.option("--remove", is_flag=True, default=False, help="Unpin instead of pin")
@click.option("--list", "list_", is_flag=True, default=False, help="List every agent's stored snapshots")
def pin_cmd(run_dir: str, snapshot: str | None, remove: bool, list_: bool) -> None:
    """Pin a stored snapshot (AGENT@vN) of a run dir so retention never evicts it, unpin it (--remove) or list
    the snapshots (--list). Safe while the run trains: its next retention pass sees the pin.

    Exit code: 0 done, 1 config error (not a run dir, unknown agent or snapshot, a snapshot the run evicted while
    it was being pinned), 2 bad arguments.
    """
    from colosseum.coordinator.checkpoint_manager import list_snapshots, parse_snapshot_ref, pin_snapshot
    from colosseum.core.errors import ConfigError

    if list_ == (snapshot is not None):
        raise click.UsageError("give either a snapshot (AGENT@vN) or --list")
    if list_ and remove:
        raise click.UsageError("--remove needs a snapshot, not --list")
    if snapshot is not None:
        try:
            agent_id, version = parse_snapshot_ref(snapshot)
        except ConfigError as e:  # a malformed reference is a usage error
            raise click.BadParameter(str(e), param_hint="'SNAPSHOT'") from e
    with _config_errors():
        if list_:
            for agent_id, infos in list_snapshots(run_dir).items():
                names = [f"{c.checkpoint_id} (pinned)" if c.meta.get("pinned") is True else c.checkpoint_id
                         for c in infos]
                click.echo(f"{agent_id}: {', '.join(names)}")
            return
        path = pin_snapshot(run_dir, agent_id, version, pinned=not remove)
    click.echo(f"{'Unpinned' if remove else 'Pinned'} {agent_id}@v{version} ({path})")


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
