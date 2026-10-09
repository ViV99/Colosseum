"""`python -m colosseum eval`: .pt and checkpoint-dir agents, roles and signatures, layouts, JSON (T6.1)."""
from __future__ import annotations

import json
import os
import time

import pytest
import torch
from click.testing import CliRunner

from colosseum.cli import main
from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.core.registry import build_model, env_spec
from colosseum.core.roles import role_signature
from colosseum.eval import load_eval_model
from game_helpers import agent_role_of, make_test_config, write_test_config

WIDE = {"model_class": "game_helpers.GameTestModel", "kwargs": {"core": "none", "hidden": 32}}


def numpy_state(model) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def _setup(tmp_path, game="turns", agent="agent_0", ckpt_agent="agent_b"):
    """Config file, config, a .pt of ``agent`` (config networks) and a checkpoint dir of a wider model."""
    cfg_path = write_test_config(tmp_path / "cfg.yaml", game)
    cfg = make_test_config(game)
    roles, role = agent_role_of(cfg, agent)
    torch.manual_seed(0)
    pt_path = tmp_path / "a.pt"
    torch.save(build_model(cfg.get_agent_config(agent), role).state_dict(), pt_path)
    wide_cfg = make_test_config(game, networks=WIDE)
    wide = build_model(wide_cfg.get_agent_config(agent), role)
    ckpt_id = CheckpointManager(tmp_path / "checkpoints").save(
        ckpt_agent, 3, numpy_state(wide),
        meta_extra={"networks": WIDE, "roles": roles, "role_signature": role_signature(role)},
    )
    return cfg_path, cfg, pt_path, tmp_path / "checkpoints" / ckpt_agent / ckpt_id


def invoke(*args):
    return CliRunner().invoke(main, ["eval", *map(str, args)])


def only_layout(cfg) -> str:
    return next(iter(env_spec(cfg).layouts))


def test_load_eval_model_builds_from_meta_networks_and_roles(tmp_path):
    _cfg_path, cfg, pt_path, ckpt_dir = _setup(tmp_path)
    model_pt, roles_pt = load_eval_model(cfg, "agent_0", pt_path)
    model_ckpt, roles_ckpt = load_eval_model(cfg, "b", ckpt_dir)
    assert roles_pt == roles_ckpt
    assert sum(p.numel() for p in model_ckpt.parameters()) > sum(p.numel() for p in model_pt.parameters())


def test_eval_cli_pairs_pt_and_checkpoint(tmp_path):
    cfg_path, cfg, pt_path, ckpt_dir = _setup(tmp_path)
    out = tmp_path / "result.json"
    result = invoke("-c", cfg_path, "-a", f"a={pt_path}", "-a", f"b={ckpt_dir}", "-n", 4, "--num-envs", 2,
                    "--seed", 0, "--output", out)
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    section = data["layouts"][only_layout(cfg)]
    assert data["num_matches"] == 4 and section["outcome_kind"] == "wdl"
    assert [(r["agent_a"], r["agent_b"]) for r in section["pairs"]] == [("a", "b"), ("b", "a")]
    assert section["pairs"][0]["n"] == 4 and set(section["pairs"][0]["per_side"]) == {"0", "1"}
    assert "win_rate" in result.output


def test_eval_cli_rejects_bad_agent_specs(tmp_path):
    cfg_path, _cfg, pt_path, _ckpt = _setup(tmp_path)
    assert invoke("-c", cfg_path, "-a", "no-equals-sign").exit_code == 2
    assert invoke("-c", cfg_path, "-a", f"a={tmp_path / 'missing.pt'}").exit_code == 2
    result = invoke("-c", cfg_path, "-a", f"a={pt_path}", "-a", f"a={pt_path}")
    assert result.exit_code == 2 and "duplicate" in result.output


def test_eval_cli_rounds_an_odd_pairwise_count_up_and_keeps_it_for_one_agent(tmp_path):
    cfg_path, cfg, pt_path, ckpt_dir = _setup(tmp_path)
    out = tmp_path / "r.json"
    result = invoke("-c", cfg_path, "-a", f"a={pt_path}", "-a", f"b={ckpt_dir}", "-n", 3, "--num-envs", 2, "-o", out)
    assert result.exit_code == 0, result.output
    assert "using 4" in result.stderr and json.loads(out.read_text())["num_matches"] == 4
    result = invoke("-c", cfg_path, "-a", f"a={pt_path}", "-n", 3, "--num-envs", 2, "-o", out)
    assert result.exit_code == 0, result.output
    section = json.loads(out.read_text())["layouts"][only_layout(cfg)]
    assert section["pairs"] == [] and section["solo"][0]["agent"] == "a" and section["n"] == 3


def test_malformed_checkpoint_and_role_signature_problems_are_config_errors(tmp_path):
    cfg_path, _cfg, pt_path, ckpt_dir = _setup(tmp_path)
    meta = json.loads((ckpt_dir / "meta.json").read_text())

    def check(new_meta: dict, message: str) -> None:
        (ckpt_dir / "meta.json").write_text(json.dumps(new_meta))
        result = invoke("-c", cfg_path, "-a", f"a={pt_path}", "-a", f"b={ckpt_dir}", "-n", 2)
        assert result.exit_code == 1, result.output
        assert result.stderr.startswith("Config error:") and message in result.stderr, result.stderr
        assert "Traceback" not in result.output

    check({**meta, "policy_version": 7}, "policy_version")
    check({**meta, "role_signature": "another-game"}, "role signature")
    check({k: v for k, v in meta.items() if k not in ("roles", "role_signature")}, "not an SP2 checkpoint")
    check({**meta, "roles": ["nope"]}, "not roles of the game")
    check({**meta, "networks": {"bogus_key": 1}}, "networks")


def test_weights_of_another_architecture_and_unusable_pt_are_config_errors(tmp_path):
    cfg_path, _cfg, _pt, ckpt_dir = _setup(tmp_path)
    wide_pt = tmp_path / "wide.pt"
    torch.save(torch.load(ckpt_dir / "model.pt", weights_only=True), wide_pt)
    result = invoke("-c", cfg_path, "-a", f"a={wide_pt}")
    assert result.exit_code == 1 and "do not match" in result.stderr
    garbage = tmp_path / "bad.pt"
    garbage.write_bytes(b"not a torch file")
    result = invoke("-c", cfg_path, "-a", f"a={garbage}")
    assert result.exit_code == 1 and result.stderr.startswith("Config error:") and str(garbage) in result.stderr


def test_load_eval_model_never_modifies_the_run(tmp_path):
    _cfg_path, cfg, _pt, ckpt_dir = _setup(tmp_path)
    agent_dir = ckpt_dir.parent
    stale = agent_dir / ".tmp-ckpt_v9-0123abcd"
    stale.mkdir()
    old = time.time() - 10 * 3600
    os.utime(stale, (old, old))
    interrupted = agent_dir / ".tmp-old-ckpt_v5-0123abcd"
    interrupted.mkdir()
    before = sorted(p.name for p in agent_dir.iterdir())
    load_eval_model(cfg, "b", ckpt_dir)
    assert sorted(p.name for p in agent_dir.iterdir()) == before


def test_layout_selection(tmp_path):
    cfg_path, _cfg, pt_path, ckpt_dir = _setup(tmp_path, game="ffa")
    out = tmp_path / "r.json"
    result = invoke("-c", cfg_path, "-a", f"a={pt_path}", "-a", f"b={ckpt_dir}", "--layout", "2p", "-n", 2,
                    "--num-envs", 2, "-o", out)
    assert result.exit_code == 0, result.output
    assert list(json.loads(out.read_text())["layouts"]) == ["2p"]
    result = invoke("-c", cfg_path, "-a", f"a={pt_path}", "--layout", "9p", "-n", 2)
    assert result.exit_code == 1 and "9p" in result.stderr


def test_asymmetric_agents_are_evaluated_in_their_roles(tmp_path):
    cfg_path, cfg, pt_path, ckpt_dir = _setup(tmp_path, game="asymmetric", agent="hunter", ckpt_agent="prey_ckpt")
    _roles, prey_role = agent_role_of(cfg, "prey")
    torch.manual_seed(1)
    prey_dir = tmp_path / "prey_ckpt"
    CheckpointManager(prey_dir).save(
        "prey", 1, numpy_state(build_model(cfg.get_agent_config("prey"), prey_role)),
        meta_extra={"roles": ["prey"], "role_signature": role_signature(prey_role)},
    )
    out = tmp_path / "r.json"
    result = invoke("-c", cfg_path, "-a", f"hunter={pt_path}", "-a", f"prey={prey_dir / 'prey' / 'ckpt_v1'}",
                    "-n", 2, "--num-envs", 2, "-o", out)
    assert result.exit_code == 0, result.output
    sections = json.loads(out.read_text())["layouts"]
    assert sections and all(set(s["by_role"]) == {"hunter", "prey"} for s in sections.values())


def test_output_dir_must_exist_before_loading(tmp_path, monkeypatch):
    import colosseum.eval as eval_module

    cfg_path, _cfg, pt_path, _ckpt = _setup(tmp_path)

    def _never(*_args, **_kwargs):
        raise AssertionError("a model was loaded before --output was checked")

    monkeypatch.setattr(eval_module, "load_eval_model", _never)
    result = invoke("-c", cfg_path, "-a", f"a={pt_path}", "-o", tmp_path / "no_such_dir" / "r.json")
    assert result.exit_code == 2 and "no_such_dir" in result.output


def test_pt_that_is_not_a_state_dict_is_a_config_error(tmp_path):
    cfg_path, _cfg, _pt, _ckpt = _setup(tmp_path)
    path = tmp_path / "list.pt"
    torch.save([torch.zeros(2)], path)
    result = invoke("-c", cfg_path, "-a", f"a={path}")
    assert result.exit_code == 1 and str(path) in result.stderr


def test_pt_of_an_unconfigured_name_needs_roles_with_one_signature(tmp_path):
    cfg_path, _cfg, pt_path, _ckpt = _setup(tmp_path, game="asymmetric", agent="hunter", ckpt_agent="prey_ckpt")
    result = invoke("-c", cfg_path, "-a", f"z={pt_path}", "-n", 2)
    assert result.exit_code == 1 and result.stderr.startswith("Config error:"), result.output
    assert str(pt_path) in result.stderr and "different spaces" in result.stderr


def test_checkpoint_networks_that_cannot_build_a_model_are_a_config_error(tmp_path):
    cfg_path, _cfg, _pt, ckpt_dir = _setup(tmp_path)
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    bad = {**WIDE, "kwargs": {**WIDE["kwargs"], "no_such_kwarg": 1}}
    (ckpt_dir / "meta.json").write_text(json.dumps({**meta, "networks": bad}))
    result = invoke("-c", cfg_path, "-a", f"b={ckpt_dir}", "-n", 1)
    assert result.exit_code == 1 and result.stderr.startswith("Config error:"), result.output
    assert str(ckpt_dir) in result.stderr and "TypeError" in result.stderr


def test_every_role_of_a_checkpoint_must_keep_its_spaces(tmp_path):
    from gymnasium.spaces import Discrete

    from colosseum.core.errors import ConfigError
    from colosseum.envs.game import GameSpec, RoleSpec, SeatSpec

    cfg = make_test_config("turns")
    _roles, role = agent_role_of(cfg, "agent_0")
    ckpt_id = CheckpointManager(tmp_path / "ck").save(
        "agent_x", 1, numpy_state(build_model(cfg.get_agent_config("agent_0"), role)),
        meta_extra={"roles": ["x", "y"], "role_signature": role_signature(role)},
    )
    ckpt_dir = tmp_path / "ck" / "agent_x" / ckpt_id

    def game(y_role):
        return GameSpec(roles={"x": role, "y": y_role}, layouts={"xy": (SeatSpec("x", 0), SeatSpec("y", 1))})

    _model, roles = load_eval_model(cfg, "b", ckpt_dir, spec=game(role))
    assert roles == ["x", "y"]
    changed = RoleSpec(role.observation_space, Discrete(5))
    with pytest.raises(ConfigError, match=r"role 'y'") as info:
        load_eval_model(cfg, "b", ckpt_dir, spec=game(changed))
    assert str(ckpt_dir) in str(info.value)


def test_each_distinct_architecture_is_validated_once(tmp_path, monkeypatch):
    import colosseum.eval as eval_module

    cfg_path, cfg, pt_path, ckpt_dir = _setup(tmp_path)
    checked = []
    real_check = eval_module._check_model
    monkeypatch.setattr(eval_module, "_check_model",
                        lambda model, role, sample, where: (checked.append(where), real_check(model, role, sample,
                                                                                              where)))
    seen: set[str] = set()
    load_eval_model(cfg, "a", pt_path, validated=seen)
    load_eval_model(cfg, "b", ckpt_dir, validated=seen)
    load_eval_model(cfg, "c", ckpt_dir, validated=seen)
    load_eval_model(cfg, "d", pt_path, validated=seen)
    assert checked == ["agent 'a'", "agent 'b'"]
    checked.clear()
    result = invoke("-c", cfg_path, "-a", f"a={pt_path}", "-a", f"b={ckpt_dir}", "-a", f"c={ckpt_dir}", "-n", 2,
                    "--num-envs", 2)
    assert result.exit_code == 0, result.output
    assert checked == ["agent 'a'", "agent 'b'"]


def test_an_architecture_that_builds_but_cannot_step_is_a_config_error(tmp_path, monkeypatch):
    from colosseum.core.errors import ConfigError
    from game_helpers import GameTestModel

    _cfg_path, cfg, pt_path, ckpt_dir = _setup(tmp_path)

    def broken_step(self, obs, state, action_mask=None):
        raise RuntimeError("broken step")

    monkeypatch.setattr(GameTestModel, "step", broken_step)
    with pytest.raises(ConfigError, match="broken step") as info:
        load_eval_model(cfg, "b", ckpt_dir)
    assert str(info.value).startswith(f"Checkpoint {ckpt_dir}: meta.json networks: agent 'b': model.step failed")
    with pytest.raises(ConfigError, match="broken step") as info:
        load_eval_model(cfg, "a", pt_path)
    assert str(info.value).startswith(f"{pt_path}: networks: agent 'a'")


def test_checkpoint_without_networks_uses_the_config_networks(tmp_path):
    cfg_path, cfg, _pt, wide_dir = _setup(tmp_path)
    roles, role = agent_role_of(cfg, "agent_0")
    ckpt_id = CheckpointManager(tmp_path / "plain").save(
        "agent_p", 1, numpy_state(build_model(cfg.get_agent_config("agent_0"), role)),
        meta_extra={"roles": roles, "role_signature": role_signature(role)},
    )
    result = invoke("-c", cfg_path, "-a", f"p={tmp_path / 'plain' / 'agent_p' / ckpt_id}", "-n", 1)
    assert result.exit_code == 0, result.output
    meta = json.loads((wide_dir / "meta.json").read_text())
    (wide_dir / "meta.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "networks"}))
    result = invoke("-c", cfg_path, "-a", f"w={wide_dir}", "-n", 1)
    assert result.exit_code == 1 and "do not match" in result.stderr


def test_eval_cli_deterministic_flag_and_repeated_layouts(tmp_path):
    cfg_path, cfg, pt_path, ckpt_dir = _setup(tmp_path)
    out = tmp_path / "r.json"
    layout = only_layout(cfg)
    result = invoke("-c", cfg_path, "-a", f"a={pt_path}", "-a", f"b={ckpt_dir}", "--layout", layout,
                    "--layout", layout, "-n", 2, "--num-envs", 2, "--deterministic", "-o", out)
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert data["deterministic"] is True and data["layouts"][layout]["n"] == 2  # the layout runs once
