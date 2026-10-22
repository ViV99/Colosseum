"""In-place resume of a run dir (SP5 spec block 5.4; task T4.3): ``colosseum hub --resume RUN`` and
``colosseum train --resume RUN`` continue the SAME dir and run id.

``prepare_resume`` runs before anything is written:
- the run dir must exist with ``config.resolved.yaml`` and ``checkpoints/``; it is refused (``ConfigError``) while
  ``run_state.json`` names a live process (two hubs on one run dir are impossible);
- the config: without ``-c`` the saved ``config.resolved.yaml`` (its SP2-SP4 ``transport`` section dropped before
  ``--set``, amendment A29; plus the agent set of ``agents_state.json`` once T8.2 writes it); with ``-c`` the new
  file, and a WARNING lists every setting that differs from the saved one; ``--set`` applies in both cases (with
  ``-c`` to the new file only, so the WARNING lists the ``--set`` changes too);
  ``run.dir`` / ``run.name`` always name the resumed dir (a ``-c`` file that names another one gets a WARNING);
- ``training.resume_from`` of the config is not applied again (the run imported its parent when it started; an
  in-place resume continues the run's own state): an INFO line, or a WARNING when ``-c`` names another source;
- the start env-step count: ``hub_state.json`` when present (an SP5 run dir), else (an SP4 / SP3 run dir) the
  largest count found in ``metrics.jsonl``, the journal, the snapshots' ``meta.json`` and ``ratings.json``;
  resuming at or past ``training.total_timesteps`` is a WARNING (spec 4.4).

``check_resumed_agents`` is the strict check of the remaining agents against the stored snapshots (run by the
caller after ``setup_run``, still before anything is written). ``commit_resume`` then keeps the previous
``config.resolved.yaml`` as ``config.resolved.<k>.yaml`` (k = 1, 2, ..., the next free number) and writes the
effective one. Everything else (the journal migration, the league replay, the hub counters, the snapshot pool, the
learners' start points) is restored by ``colosseum.hub.hub.Hub``.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from colosseum.core.config import ColosseumConfig, apply_overrides, load_config
from colosseum.core.errors import ConfigError
from colosseum.core.run_dir import RESOLVED_CONFIG_FILE, RunDir, read_run_id
from colosseum.core.run_state import process_alive, read_run_state
from colosseum.hub.state import HUB_STATE_FILE, load_hub_state

logger = logging.getLogger(__name__)

__all__ = [
    "AGENTS_STATE_FILE", "DIFF_LINES_MAX", "ResumePlan", "check_not_running", "check_resumed_agents", "commit_resume",
    "config_diff", "flatten", "next_backup_path", "prepare_resume", "resume_port", "start_env_steps",
]

AGENTS_STATE_FILE = "agents_state.json"      # written by T8.2 (agents added / removed on the fly)
_BACKUP_RE = re.compile(r"config\.resolved\.(\d+)\.yaml")
DIFF_LINES_MAX = 40                          # diff lines of the -c WARNING (the rest is counted)
_PINNED_KEYS = ("run.dir", "run.name")


@dataclass(frozen=True)
class ResumePlan:
    run_dir: RunDir
    config: ColosseumConfig
    from_file: bool                  # -c given (False: the saved config)
    diff: list[str]                  # settings that differ from the saved config ("key: old -> new")
    run_id: str | None               # None for an SP3 run dir (no run.json yet; the hub creates one)
    sp5: bool                        # hub_state.json present
    start_env_steps: int


def check_not_running(root: str | Path) -> None:
    """ConfigError while ``run_state.json`` of ``root`` names a live process (spec 5.4)."""
    state = read_run_state(root)
    if state is not None and state.status == "running" and process_alive(state):
        raise ConfigError(f"Run {root} is running (pid {state.pid}); stop it first (Ctrl+C in its terminal) or "
                          f"wait until it exits: two hubs cannot share one run dir")


def flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    """Dotted keys of a nested dict (lists and empty dicts are leaves)."""
    if not isinstance(data, Mapping):
        return {prefix: data}
    out: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping) and value:
            out.update(flatten(value, path))
        else:
            out[path] = value
    return out


def config_diff(old: ColosseumConfig, new: ColosseumConfig) -> list[str]:
    """Settings that differ between two resolved configs, sorted by key: ``key: old -> new``, ``key: added (v)``,
    ``key: removed (was v)`` (``run.dir`` / ``run.name`` are pinned to the run dir and left out)."""
    a = flatten(old.model_dump(mode="json", by_alias=True))
    b = flatten(new.model_dump(mode="json", by_alias=True))
    keys = set(a) | set(b)
    lines = []
    for key in sorted(keys):
        if key in _PINNED_KEYS:
            continue
        if (a.get(key, b.get(key)) == {} or b.get(key, a.get(key)) == {}) and any(
                k.startswith(f"{key}.") for k in keys):
            continue                                   # an empty section on one side, its keys on the other
        if key not in b:
            lines.append(f"{key}: removed (was {a[key]!r})")
        elif key not in a:
            lines.append(f"{key}: added ({b[key]!r})")
        elif a[key] != b[key]:
            lines.append(f"{key}: {a[key]!r} -> {b[key]!r}")
    return lines


def next_backup_path(root: str | Path) -> Path:
    """``config.resolved.<k>.yaml`` with the next free k (1 for the first resume)."""
    root = Path(root)
    taken = [int(m.group(1)) for p in root.glob("config.resolved.*.yaml") if (m := _BACKUP_RE.fullmatch(p.name))]
    return root / f"config.resolved.{max(taken, default=0) + 1}.yaml"


def _pin_run(config: ColosseumConfig, root: Path, *, warn: bool) -> ColosseumConfig:
    """``config`` with ``run.dir`` / ``run.name`` naming ``root`` (a WARNING with ``warn`` when it named another
    dir)."""
    run = config.run
    if warn and ((run.name is not None and run.name != root.name)
                 or Path(run.dir).resolve() != root.parent.resolve()):
        logger.warning(f"Resume: the config names run dir {Path(run.dir) / str(run.name)}; the resumed run stays in "
                       f"{root}")
    return config.model_copy(update={"run": run.model_copy(update={"dir": str(root.parent), "name": root.name})})


def _apply_agents_state(config: ColosseumConfig, root: Path) -> ColosseumConfig:
    """The agent set of ``agents_state.json`` (T8.2 fills this in; until then the file does not exist)."""
    return config


def _load_saved(root: Path, overrides: Mapping[str, Any] | None) -> ColosseumConfig:
    """The saved ``config.resolved.yaml`` with ``overrides`` (``--set``). Its ``transport`` section (SP2-SP4 resolved
    configs keep it) is dropped before ``--set`` (A29): SP5 reads ``distributed.*``, and translating it again would
    WARN on every resume and conflict with ``--set distributed.max_message_mb``."""
    path = root / RESOLVED_CONFIG_FILE
    if not path.is_file():
        raise ConfigError(f"{root} has no {RESOLVED_CONFIG_FILE}: it is not a run dir (resume needs the dir "
                          f"colosseum train / hub created, e.g. runs/<name>)")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"Cannot read config {path}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"Cannot read config {path}: not a mapping")
    raw.pop("transport", None)
    if overrides:
        raw = apply_overrides(raw, dict(overrides))
    try:
        return ColosseumConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"Invalid config {path}:\n{e}") from e


def start_env_steps(root: str | Path) -> int:
    """The env-step count an SP4 / SP3 run dir reached (no ``hub_state.json``): the largest of ``metrics.jsonl``
    (any record's ``env_steps``; a torn last line is skipped), the journal's training matches, the snapshots'
    ``meta.json`` and ``ratings.json`` (0 when none has one). Reads only."""
    root = Path(root)
    found = [0]
    metrics = root / "metrics.jsonl"
    if metrics.is_file():
        with metrics.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    value = json.loads(line).get("env_steps")
                except (ValueError, AttributeError):
                    continue                       # a torn last line of a killed run
                if isinstance(value, int):
                    found.append(value)
    journal = root / "matches.db"
    if journal.is_file():
        from colosseum.arena.matchlog import MatchLogError, MatchLogReader

        try:
            with MatchLogReader(journal) as reader:
                found.append(reader.max_env_steps() or 0)
        except MatchLogError as e:
            logger.warning(f"Resume: the journal's env steps could not be read ({e})")
    for meta in (root / "checkpoints").glob("*/ckpt_v*/meta.json"):
        try:
            value = json.loads(meta.read_text()).get("env_steps")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(value, int):
            found.append(value)
    try:
        value = json.loads((root / "ratings.json").read_text()).get("env_steps")
        if isinstance(value, int):
            found.append(value)
    except (OSError, ValueError, AttributeError):
        pass
    return max(found)


def _hub_state_env_steps(root: Path) -> int | None:
    """``hub_state.json``'s env-step count (None: no file, an SP4 / SP3 run dir); a malformed file is a ConfigError
    (the hub writes it atomically)."""
    path = root / HUB_STATE_FILE
    try:
        data = load_hub_state(path)
    except ValueError as e:
        raise ConfigError(f"Cannot read {path} ({e}); the hub writes it atomically, so it was edited: restore it, or "
                          f"move it away to continue with the counters of metrics.jsonl and the journal") from e
    return None if data is None else int(data.env_steps)


def prepare_resume(run: str | Path, config_path: str | Path | None,
                   overrides: Mapping[str, Any] | None = None) -> ResumePlan:
    """Check and load an in-place resume (module docstring). Nothing is written."""
    root = Path(run)
    if not root.is_dir() or not (root / "checkpoints").is_dir():
        raise ConfigError(f"--resume {run}: not a run dir (expected a dir with checkpoints/ and "
                          f"{RESOLVED_CONFIG_FILE}, e.g. runs/<name>)")
    check_not_running(root)
    if config_path is None:
        saved = _apply_agents_state(_load_saved(root, overrides), root)
        config, diff = saved, []
    else:
        # With -c, --set applies only to the new config: the saved one is the reference of the diff (which then
        # lists the --set changes too) and of the resume_from comparison; an override that names an agent only the
        # new file has must not reach the saved config (review round 1).
        saved = _apply_agents_state(_load_saved(root, None), root)
        config = load_config(config_path, dict(overrides) if overrides else None)
        diff = config_diff(_pin_run(saved, root, warn=False), _pin_run(config, root, warn=False))
        if diff:
            shown = diff[:DIFF_LINES_MAX]
            more = f"\n  ... and {len(diff) - len(shown)} more" if len(diff) > len(shown) else ""
            logger.warning(f"Resume: {config_path} differs from the run's saved config in {len(diff)} settings:\n  "
                           + "\n  ".join(shown) + more)
    config = _pin_run(config, root, warn=config_path is not None)
    if config.training.resume_from:
        if config_path is not None and config.training.resume_from != saved.training.resume_from:
            logger.warning(f"Resume: training.resume_from={config.training.resume_from!r} is ignored: an in-place "
                           f"resume continues the run's own state (start a new run to resume from another one)")
        else:
            logger.info(f"Resume: training.resume_from={config.training.resume_from!r} was imported when the run "
                        f"started; the run continues from its own state")
    hub_steps = _hub_state_env_steps(root)
    steps = hub_steps if hub_steps is not None else start_env_steps(root)
    if steps >= config.training.total_timesteps:
        logger.warning(f"Resume: the run is at {steps} env steps, at or past training.total_timesteps "
                       f"({config.training.total_timesteps}); it stops at once unless you raise it "
                       f"(--set training.total_timesteps=N)")
    return ResumePlan(run_dir=RunDir(root, root.name), config=config, from_file=config_path is not None, diff=diff,
                      run_id=read_run_id(root), sp5=hub_steps is not None, start_env_steps=steps)


def commit_resume(plan: ResumePlan) -> Path | None:
    """Keep the previous ``config.resolved.yaml`` as ``config.resolved.<k>.yaml`` and write the effective one;
    returns the backup path (None when there was nothing to keep)."""
    root = plan.run_dir.root
    current = root / RESOLVED_CONFIG_FILE
    backup = None
    if current.is_file():
        backup = next_backup_path(root)
        shutil.copy2(current, backup)
    plan.run_dir.write_resolved_config(plan.config)
    return backup


def check_resumed_agents(root: str | Path, agent_configs: Mapping[str, ColosseumConfig],
                         role_specs: Mapping[str, Any]) -> dict[str, str]:
    """Strict checks of an in-place resume (spec 5.4), before anything is written: for every trainable agent of
    ``agent_configs`` (``setup_run``'s per-agent configs) that has snapshots in the run dir, the newest snapshot's
    role signature must be that of the agent's roles in this config and its weights must load into the agent's
    model (as a strict ``training.resume_from``); ConfigError naming the agent and the snapshot otherwise. An agent
    without snapshots (added with ``-c``) starts fresh. Returns ``{agent: checked snapshot}``."""
    from colosseum.coordinator.checkpoint_manager import check_model_state, resolve_resume
    from colosseum.core.registry import build_model
    from colosseum.core.roles import role_signature

    checked: dict[str, str] = {}
    for aid, acfg in agent_configs.items():
        if not (Path(root) / "checkpoints" / aid).is_dir():
            continue
        try:
            state = resolve_resume(str(root), aid, expected_signature=role_signature(role_specs[aid]))
            if state is None:
                continue
            check_model_state(build_model(acfg, role_specs[aid]), state["model_state"], state["source"])
        except ConfigError as e:
            raise ConfigError(f"--resume {root}: agent {aid!r} cannot continue with this config: {e}") from e
        checked[aid] = state["source"]
    return checked


def resume_port(previous_addresses: Sequence[str], port: int | None, config_port: int) -> int:
    """The port a resumed ``colosseum hub`` listens on: ``--port`` when given; else the one port of the previous
    incarnation's addresses (``hub.json``), so nodes reconnect to the address they know (ruling W04-20); else
    ``distributed.port``."""
    if port is not None:
        return int(port)
    ports = {address.rsplit(":", 1)[1] for address in previous_addresses if ":" in address}
    if len(ports) == 1:
        found = ports.pop()
        if found.isdigit():
            return int(found)
    return int(config_port)
