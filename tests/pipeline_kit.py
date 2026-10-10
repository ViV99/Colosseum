"""The SP3 competition pipeline as test helpers: ``record`` -> ``bc`` -> ``train`` (``init`` with
critic warm-up, kickstart, anchors) -> ``eval`` (spec section 3, criterion 3).

Used by the fast tic-tac-toe smoke (``tests/integration/test_sp3_pipeline_smoke.py``), the slow
unit_harvest test (``tests/learning/test_sp3_pipeline_slow.py``) and
``scripts/pipeline_vs_scratch.py``. Every step is the real CLI: in a subprocess through
``cli_runner.run_in_session`` (default) or in-process through click's ``CliRunner``
(``in_process=True``); ``train`` always runs in a subprocess (it starts processes).

The pipeline config is an example config with these agents (``pipeline_config``):
- ``main``: the trainable agent (``init`` and ``kickstart`` are added for the train step);
- ``PipelineGame.bot``: the game's scripted bot (recorded, an anchor, possibly the DAgger teacher);
- ``random``: ``colosseum.players.RandomBot``;
- ``bc_net``: the frozen BC weights, once they exist.
SP2 matchmaking knobs are dropped and ``pool_size`` becomes ``keep_last``, so old and new knobs
never meet (that would be a ConfigError).
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from cli_runner import REPO_ROOT, TrainRun, run_in_session, run_train

MAIN = "main"
RANDOM = "random"
BC_NET = "bc_net"
TRAINED = "trained"
RANDOM_BOT_CLASS = "colosseum.players.RandomBot"
SP2_MATCHMAKING_KNOBS = ("mode", "self_play_ratio", "latest_prob", "pfsp_exponent")
_CKPT_RE = re.compile(r"ckpt_v(\d+)")

# Short training of the fast smoke (cli_runner.TINY, with keep_last instead of SP2's pool_size).
SMOKE_SETTINGS: dict[str, Any] = {
    "training.total_timesteps": 3000, "rollout.num_workers": 1, "rollout.envs_per_worker": 8,
    "rollout.chunk_length": 16, "rollout.weight_sync_interval_sec": 0.5, "rollout.match_refresh_interval_sec": 1.0,
    "learner.batch_chunks": 2, "learner.queue_size": 16, "checkpoint.interval": 20, "checkpoint.keep_last": 5,
    "metrics.log_interval": 1, "metrics.console_interval_sec": 1.0,
}


@dataclass(frozen=True)
class PipelineGame:
    config: str                                     # configs/examples/<config>.yaml
    bot: str                                        # agent id of the game's bot in the pipeline config
    bot_class: str                                  # its ScriptedBot (added when the config has no such agent)
    layout: str                                     # the layout whose eval rows are returned
    opponents: Mapping[str, float] | None = None    # matchmaking.opponents (None: the config's own)


@dataclass(frozen=True)
class PipelineSettings:
    record_matches: int
    bc_epochs: int
    critic_warmup_steps: int
    kickstart_teacher: str                          # an agent id (scripted: DAgger labels; frozen: KL)
    kickstart_lambda: float
    kickstart_decay_steps: int
    eval_matches: int
    train_sets: Mapping[str, Any] = field(default_factory=dict)   # --set of the train step
    init_from: str = BC_NET                         # BC_NET (the frozen agent's name) or "path" (bc.pt)


TIC_TAC_TOE = PipelineGame("tic_tac_toe", "ttt_bot", "examples.tic_tac_toe.bots.TicTacToeBot", "2p",
                           opponents={"latest": 0.4, "snapshots": 0.1, "rivals": 0.0, "anchors": 0.5})
UNIT_HARVEST = PipelineGame("unit_harvest_league", "greedy", "examples.unit_harvest.bots.HarvestBot", "2p")

TIC_TAC_TOE_SMOKE = PipelineSettings(record_matches=20, bc_epochs=2, critic_warmup_steps=2, kickstart_teacher=BC_NET,
                                     kickstart_lambda=1.0, kickstart_decay_steps=50, eval_matches=4,
                                     train_sets=SMOKE_SETTINGS)
# T6.4 calibration (ruling): budgets of the slow unit_harvest pipeline test.
UNIT_HARVEST_SETTINGS = PipelineSettings(record_matches=300, bc_epochs=5, critic_warmup_steps=30,
                                         kickstart_teacher="greedy", kickstart_lambda=1.0, kickstart_decay_steps=300,
                                         eval_matches=60, train_sets={"rollout.num_workers": 2}, init_from="path")


def pipeline_config(game: PipelineGame, *, bc_path: Path | None = None, init: Mapping[str, Any] | None = None,
                    kickstart: Mapping[str, Any] | None = None) -> dict:
    """The raw pipeline config (module docstring)."""
    raw = yaml.safe_load((REPO_ROOT / "configs" / "examples" / f"{game.config}.yaml").read_text())
    agents = {name: dict(entry or {}) for name, entry in (raw.get("agents") or {}).items()}
    agents.setdefault(MAIN, {})
    agents.setdefault(game.bot, {"kind": "scripted", "class": game.bot_class})
    agents.setdefault(RANDOM, {"kind": "scripted", "class": RANDOM_BOT_CLASS})
    if bc_path is not None:
        agents[BC_NET] = {"kind": "frozen", "path": str(bc_path)}
    if init is not None:
        agents[MAIN]["init"] = dict(init)
    if kickstart is not None:
        agents[MAIN]["kickstart"] = dict(kickstart)
    raw["agents"] = agents
    matchmaking = {k: v for k, v in (raw.get("matchmaking") or {}).items() if k not in SP2_MATCHMAKING_KNOBS}
    if game.opponents is not None:
        matchmaking["opponents"] = dict(game.opponents)
    raw["matchmaking"] = matchmaking
    checkpoint = dict(raw.get("checkpoint") or {})
    if "pool_size" in checkpoint:
        checkpoint["keep_last"] = checkpoint.pop("pool_size")
    raw["checkpoint"] = checkpoint
    return raw


def write_config(path: Path, raw: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(raw), sort_keys=False))
    return path


@dataclass
class CliResult:
    code: int
    stdout: str
    stderr: str


def cli(args: Sequence[Any], *, in_process: bool = False, timeout: float = 900.0) -> CliResult:
    """``colosseum <args>``: in a subprocess (own session, killed on exit) or in-process."""
    args = [str(a) for a in args]
    if in_process:
        from click.testing import CliRunner

        from colosseum.cli import main

        result = CliRunner().invoke(main, args)
        if result.exception is not None and not isinstance(result.exception, SystemExit):
            raise result.exception
        return CliResult(result.exit_code, result.stdout, result.stderr)
    proc = run_in_session([sys.executable, "-m", "colosseum", *args], timeout)
    return CliResult(proc.returncode, proc.stdout, proc.stderr)


def _ok(result: CliResult, what: str) -> CliResult:
    assert result.code == 0, (f"{what} failed with exit code {result.code}:\n{result.stdout[-2000:]}\n"
                              f"{result.stderr[-3000:]}")
    return result


def record(config: Path, player: str, out: Path, *, num_matches: int, seed: int, in_process: bool = False) -> dict:
    _ok(cli(["record", "-c", config, "--player", player, "--num-matches", num_matches, "--output", out,
             "--seed", seed], in_process=in_process), "record")
    return json.loads((Path(out) / "record.json").read_text())


def bc(config: Path, data: Path, out: Path, *, epochs: int, agent: str = MAIN, in_process: bool = False) -> Path:
    _ok(cli(["bc", "-c", config, "--agent", agent, "--data", data, "--output", out, "--epochs", epochs],
            in_process=in_process), "bc")
    return Path(out)


def train(config: Path, workdir: Path, name: str, sets: Mapping[str, Any], *, timeout: float = 900.0) -> TrainRun:
    """``colosseum train`` into ``<workdir>/runs/<name>`` with the config's own sizes plus ``sets``."""
    run = run_train(config, workdir, name, dict(sets), timeout=timeout, tiny=False)
    assert run.returncode == 0, run.stderr[-3000:]
    return run


def newest_checkpoint(root: Path, agent_id: str) -> Path:
    agent_dir = Path(root) / "checkpoints" / agent_id
    versions = {int(m.group(1)): d for d in agent_dir.iterdir()
                if d.is_dir() and (m := _CKPT_RE.fullmatch(d.name))} if agent_dir.is_dir() else {}
    assert versions, f"no checkpoints of {agent_id} in {root}"
    return versions[max(versions)]


def evaluate(config: Path, checkpoint: Path, opponents: Sequence[str], out: Path, *, layout: str, num_matches: int,
             seed: int = 12345, in_process: bool = False) -> dict[str, dict]:
    """``colosseum eval -a trained=<checkpoint> -a <opponent> ... --deterministic`` (opponents by name);
    returns the ``wdl`` rows "trained vs <opponent>" of ``layout``, keyed by opponent."""
    args: list[Any] = ["eval", "-c", config, "-a", f"{TRAINED}={checkpoint}"]
    for name in opponents:
        args += ["-a", name]
    args += ["--layout", layout, "--num-matches", num_matches, "--seed", seed, "--deterministic", "--output", out]
    _ok(cli(args, in_process=in_process), "eval")
    report = json.loads(Path(out).read_text())
    return {row["agent_b"]: row for row in report["layouts"][layout]["pairs"] if row["agent_a"] == TRAINED}


@dataclass
class PipelineRun:
    record: dict | None              # record.json (None when bc_path was given)
    bc_path: Path
    run: TrainRun
    checkpoint: Path
    pairs: dict[str, dict]           # opponent -> eval row "trained vs opponent"
    seconds: dict[str, float]


def run_pipeline(game: PipelineGame, workdir: Path, *, seed: int, settings: PipelineSettings, warm_start: bool = True,
                 bc_path: Path | None = None, name: str = "pipeline", in_process: bool = False) -> PipelineRun:
    """record -> bc (both skipped when ``bc_path`` is given) -> train -> eval against the bot, RandomBot
    and the frozen BC net. ``warm_start=False`` trains in the same league without ``init`` and
    kickstart (the "pipeline vs scratch" comparison)."""
    workdir = Path(workdir)
    seconds: dict[str, float] = {}
    record_json = None
    if bc_path is None:
        record_cfg = write_config(workdir / "record.yaml", pipeline_config(game))
        start = time.monotonic()
        record_json = record(record_cfg, game.bot, workdir / "data", num_matches=settings.record_matches, seed=seed,
                             in_process=in_process)
        seconds["record"] = time.monotonic() - start
        start = time.monotonic()
        bc_path = bc(record_cfg, workdir / "data", workdir / "bc.pt", epochs=settings.bc_epochs, in_process=in_process)
        seconds["bc"] = time.monotonic() - start
    init = kickstart = None
    if warm_start:
        init = {"from": BC_NET if settings.init_from == BC_NET else str(bc_path),
                "critic_warmup_steps": settings.critic_warmup_steps}
        kickstart = {"teacher": settings.kickstart_teacher, "lambda": settings.kickstart_lambda,
                     "decay_steps": settings.kickstart_decay_steps}
    train_cfg = write_config(workdir / f"{name}.yaml",
                             pipeline_config(game, bc_path=bc_path, init=init, kickstart=kickstart))
    start = time.monotonic()
    run = train(train_cfg, workdir, name, {"training.seed": seed, **settings.train_sets})
    seconds["train"] = time.monotonic() - start
    checkpoint = newest_checkpoint(run.root, MAIN)
    start = time.monotonic()
    pairs = evaluate(train_cfg, checkpoint, [game.bot, RANDOM, BC_NET], workdir / f"{name}-eval.json",
                     layout=game.layout, num_matches=settings.eval_matches, in_process=in_process)
    seconds["eval"] = time.monotonic() - start
    return PipelineRun(record_json, Path(bc_path), run, checkpoint, pairs, seconds)
