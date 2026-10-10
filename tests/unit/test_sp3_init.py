"""init (spec block 6): .pt, checkpoint dir, run dir and frozen-agent sources; strict and partial
loading with a report; resume takes precedence; the learner applies it (T4.2)."""
from __future__ import annotations

import json
import logging
import sys

import numpy as np
import pytest
import torch

from colosseum.core.config import load_config
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_model, env_spec, validate_config
from colosseum.learner.factory import InitState, apply_init, resolve_init
from game_helpers import (
    SP2_CHECKPOINT,
    SP2_TTT_TINY,
    agent_role_of,
    make_test_config,
    make_test_run_dir,
    write_ckpt_dir,
)

BOT = {"kind": "scripted", "class": "colosseum.players.RandomBot"}


def _flat(state: dict) -> np.ndarray:
    return np.concatenate([np.asarray(v.detach().cpu() if hasattr(v, "detach") else v).ravel()
                           for _, v in sorted(state.items())])


def _model(core: str = "none", seed: int = 5):
    cfg = make_test_config("turns", networks={"kwargs": {"core": core, "hidden": 16}})
    torch.manual_seed(seed)
    return build_model(cfg.get_agent_config("agent_0"), agent_role_of(cfg, "agent_0")[1])


def _pt(tmp_path, model, name: str = "init.pt"):
    path = tmp_path / name
    torch.save(model.state_dict(), path)
    return path


def _resolve(config, agent_id: str = "agent_0"):
    return resolve_init(config, agent_id, env_spec(config))


def _fresh(config, agent_id: str = "agent_0"):
    torch.manual_seed(123)
    return build_model(config.get_agent_config(agent_id), agent_role_of(config, agent_id)[1])


def test_without_init_from_there_is_nothing_to_load():
    assert _resolve(make_test_config("turns")) is None
    assert _resolve(make_test_config("turns", init={"critic_warmup_steps": 3})) is None


def test_strict_pt_loads_every_tensor(tmp_path):
    source = _model()
    cfg = make_test_config("turns", init={"from": str(_pt(tmp_path, source))})
    init = _resolve(cfg)
    assert isinstance(init, InitState) and init.strict and init.source == str(tmp_path / "init.pt")
    assert len(init.report) == 1 and "(strict)" in init.report[0] and "agent 'agent_0'" in init.report[0]
    model = _fresh(cfg)
    apply_init(model, init)
    assert np.array_equal(_flat(model.state_dict()), _flat(source.state_dict()))


def test_strict_mismatch_is_a_config_error_with_a_hint(tmp_path):
    cfg = make_test_config("turns", init={"from": str(_pt(tmp_path, _model(core="lstm")))})
    with pytest.raises(ConfigError, match="strict: false"):
        _resolve(cfg)


def test_partial_loads_the_matching_tensors_and_reports_the_rest(tmp_path):
    source = _model(core="lstm")
    cfg = make_test_config("turns", init={"from": str(_pt(tmp_path, source)), "strict": False})
    init = _resolve(cfg)
    assert not init.strict and "(partial)" in init.report[0]
    assert any(line.startswith("  not in the model (ignored)") and "core" in line for line in init.report)
    model = _fresh(cfg)
    apply_init(model, init)
    for key, value in model.state_dict().items():
        assert torch.equal(value, source.state_dict()[key]), key            # every target tensor matched
    target_lstm = make_test_config("turns", networks={"kwargs": {"core": "lstm", "hidden": 16}},
                                   init={"from": str(_pt(tmp_path, _model(core="none"), "plain.pt")),
                                         "strict": False})
    report = _resolve(target_lstm).report
    assert any(line.startswith("  not in the source (kept as initialized)") and "core" in line for line in report)


def test_partial_without_any_match_is_a_config_error(tmp_path):
    cfg = make_test_config("turns", init={"from": str(_pt(tmp_path, torch.nn.Linear(2, 2))), "strict": False})
    with pytest.raises(ConfigError, match="no tensor"):
        _resolve(cfg)


def test_checkpoint_dir_with_its_role_signature(tmp_path):
    base = make_test_config("turns")
    ckpt = write_ckpt_dir(tmp_path / "old", base, seed=6)
    init = _resolve(make_test_config("turns", init={"from": str(ckpt)}))
    assert init.source == str(ckpt)
    torch.manual_seed(6)
    expected = build_model(base.get_agent_config("agent_0"), agent_role_of(base, "agent_0")[1])
    assert np.array_equal(_flat(init.model_state), _flat(expected.state_dict()))
    with pytest.raises(ConfigError, match="role signature"):          # turns has 3-d observations, simultaneous 2-d
        _resolve(make_test_config("simultaneous", init={"from": str(ckpt)}))
    meta = json.loads((ckpt / "meta.json").read_text())
    del meta["role_signature"]
    (ckpt / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ConfigError, match="no role_signature"):
        _resolve(make_test_config("turns", init={"from": str(ckpt)}))


def test_run_dir_takes_the_agents_latest_checkpoint(tmp_path):
    base = make_test_config("turns")
    agent_dir = tmp_path / "old_run" / "checkpoints" / "agent_0"
    write_ckpt_dir(agent_dir, base, policy_version=3, seed=1)
    latest = write_ckpt_dir(agent_dir, base, policy_version=8, seed=2)
    (agent_dir / ".tmp-ckpt_v9-abc").mkdir()                       # leftovers are ignored
    init = _resolve(make_test_config("turns", init={"from": str(tmp_path / "old_run")}))
    assert init.source == str(latest)
    other = make_test_config("turns", agents={"other": {}}, init={"from": str(tmp_path / "old_run")})
    with pytest.raises(ConfigError, match="no checkpoints of agent 'other'"):
        _resolve(other, "other")


def test_frozen_agent_by_name(tmp_path):
    base = make_test_config("turns")
    ckpt = write_ckpt_dir(tmp_path / "prev", base, seed=9)
    cfg = make_test_config("turns", agents={"agent_0": {"init": {"from": "prev"}},
                                            "prev": {"kind": "frozen", "path": str(ckpt)}},
                           matchmaking={"anchors": []})
    init = _resolve(cfg)
    assert "frozen agent 'prev'" in init.source and init.strict


@pytest.mark.parametrize("agents, source, message", [
    ({"agent_0": {}, "bot": BOT}, "bot", "scripted agent"),
    ({"agent_0": {}, "b": {}}, "b", "trainable agent"),
    ({"agent_0": {}}, "nowhere.pt", "neither a frozen agent"),
])
def test_bad_sources_are_config_errors(agents, source, message):
    cfg = make_test_config("turns", agents=agents, init={"from": source})
    with pytest.raises(ConfigError, match=message):
        _resolve(cfg)


def test_init_is_per_agent(tmp_path):
    path = _pt(tmp_path, _model())
    cfg = make_test_config("turns", agents={"a": {}, "b": {"init": {"from": None}}}, init={"from": str(path)})
    assert _resolve(cfg, "a") is not None and _resolve(cfg, "b") is None


def test_validate_prints_the_init_report(tmp_path):
    cfg = make_test_config("turns", init={"from": str(_pt(tmp_path, _model(core="lstm"))), "strict": False})
    lines = validate_config(cfg).lines
    assert any("init from" in line and "(partial)" in line for line in lines)
    assert any(line.startswith("  not in the model (ignored)") for line in lines)


def test_the_sp2_checkpoint_fixture_works_as_init():
    cfg = load_config(SP2_TTT_TINY, {"init.from": str(SP2_CHECKPOINT)})    # SP2-format config, translated
    init = _resolve(cfg)
    assert init.strict and init.source == str(SP2_CHECKPOINT)


def test_resume_takes_precedence_over_init(tmp_path, caplog):
    from colosseum.launcher import Launcher

    cfg = make_test_config("turns", agents={"a": {}, "b": {}}, init={"from": str(_pt(tmp_path, _model()))})
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path))
    with caplog.at_level(logging.INFO, logger="colosseum.launcher"):
        states = launcher._resolve_init(env_spec(cfg), {"a": {"source": "old/checkpoints/a/ckpt_v3"}, "b": None})
    assert states["a"] is None and isinstance(states["b"], InitState)
    assert any("'a'" in r.getMessage() and "init.from" in r.getMessage() and "ignored" in r.getMessage()
               for r in caplog.records)


def test_the_learner_starts_from_the_init_weights_at_version_zero(tmp_path, monkeypatch, restore_global_rng):
    import colosseum.learner.learner as learner_module
    from colosseum.launcher import _learner_main

    source = _model(seed=11)
    cfg = make_test_config("turns", learner={"device": "cpu", "torch_threads": 1},
                           init={"from": str(_pt(tmp_path, source))})
    built = []
    monkeypatch.setattr(learner_module, "learner_process", lambda *, algorithm_factory, **kw: built.append(
        algorithm_factory()))
    monkeypatch.setattr(sys, "path", list(sys.path))
    _roles, role = agent_role_of(cfg, "agent_0")
    before = torch.get_num_threads()
    try:
        _learner_main(agent_id="agent_0", config=cfg.get_agent_config("agent_0"), role_spec=role,
                      spec=env_spec(cfg), init_state=_resolve(cfg), trajectory_queue=None, weight_queues=[],
                      stop_event=None, metrics_queue=None)
    finally:
        torch.set_num_threads(before)
    (algo,) = built
    assert np.array_equal(_flat(algo.model.state_dict()), _flat(source.state_dict()))
    assert algo.policy_version == 0 and algo.state_dict()["optimizer"]["state"] == {}


def test_the_launcher_passes_init_states_to_learners(tmp_path, monkeypatch, restore_global_rng):
    import colosseum.launcher as launcher_module

    cfg = make_test_config("turns", init={"from": str(_pt(tmp_path, _model()))})
    learner_kwargs: list[dict] = []

    class _Stop(Exception):
        pass

    class FakeProcess:
        exitcode = 0

        def __init__(self, target, kwargs, name=None, daemon=None):
            self.target, self.kwargs = target, kwargs

        def start(self):
            if self.target is launcher_module._learner_target:
                learner_kwargs.append(self.kwargs)
            else:
                raise _Stop

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(launcher_module.mp, "Process", FakeProcess)
    with pytest.raises(_Stop):
        launcher_module.Launcher(cfg, make_test_run_dir(cfg, tmp_path)).launch()
    (kwargs,) = learner_kwargs
    assert isinstance(kwargs["init_state"], InitState) and kwargs["init_state"].strict
