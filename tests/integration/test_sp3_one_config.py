"""One config for the whole competition pipeline (spec blocks 1 and 6; fix-wave ruling A-I1/C-I1):
``record``, ``bc`` and ``eval`` validate in the "play" scope, so a config that already names the BC
output (a frozen ``bc_net`` with ``path: bc.pt``, ``init.from: bc.pt``, ``kickstart.teacher: bc_net``)
works before ``bc.pt`` exists; ``validate`` / ``train`` keep the full check. The three commands take
``--set`` like ``train``."""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from colosseum.cli import main
from colosseum.core.errors import ConfigError
from colosseum.core.registry import validate_config
from game_helpers import frozen_agent, make_test_config, scripted_agent, write_test_config

pytestmark = pytest.mark.usefixtures("restore_root_logging")


def one_config(tmp_path, bc_pt):
    agents = {
        "main": {"init": {"from": str(bc_pt), "critic_warmup_steps": 2}, "kickstart": {"teacher": "bc_net"}},
        "bot": scripted_agent(),
        "bc_net": frozen_agent(bc_pt),
    }
    return write_test_config(tmp_path / "one.yaml", "turns", agents=agents), make_test_config("turns", agents=agents)


def run(*args):
    result = CliRunner().invoke(main, [str(a) for a in args])
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result


def test_record_bc_validate_and_eval_run_from_one_config_before_bc_pt_exists(tmp_path):
    bc_pt = tmp_path / "bc.pt"
    cfg_path, cfg = one_config(tmp_path, bc_pt)
    with pytest.raises(ConfigError, match="bc_net"):
        validate_config(cfg)                                              # the full check needs bc.pt
    validate_config(cfg, scope="play", players=["bot"])                  # record --player bot
    validate_config(cfg, scope="play")                                   # bc

    rec = run("record", "-c", cfg_path, "--player", "bot", "--num-matches", 4, "--num-envs", 2, "--seed", 0,
              "--output", tmp_path / "data")
    assert rec.exit_code == 0, rec.output
    trained = run("bc", "-c", cfg_path, "--agent", "main", "--data", tmp_path / "data", "--output", bc_pt,
                  "--epochs", 1, "--seed", 0)
    assert trained.exit_code == 0, trained.output
    assert bc_pt.is_file()
    checked = run("validate", "-c", cfg_path)
    assert checked.exit_code == 0, checked.output

    moved = tmp_path / "moved.pt"
    bc_pt.rename(moved)                                                   # eval without bc.pt: bc_net not named
    evaluated = run("eval", "-c", cfg_path, "-a", "bot", "-a", f"main={moved}", "--num-matches", 2,
                    "--num-envs", 2, "--seed", 0)
    assert evaluated.exit_code == 0, evaluated.output
    named = run("eval", "-c", cfg_path, "-a", "bot", "-a", "bc_net", "--num-matches", 2, "--num-envs", 2)
    assert named.exit_code == 1 and "bc_net" in named.output              # a named frozen agent is loaded
    overridden = run("eval", "-c", cfg_path, "-a", "bot", "-a", "bc_net", "--num-matches", 2, "--num-envs", 2,
                     "--set", f"agents.bc_net.path={moved}")
    assert overridden.exit_code == 0, overridden.output


@pytest.mark.parametrize("command", [
    ["record", "--player", "bot", "--output", "{tmp}/rec"],
    ["bc", "--data", "{cfg}", "--output", "{tmp}/bc.pt"],
    ["eval", "-a", "bot"],
])
def test_record_bc_and_eval_take_set_overrides(tmp_path, command):
    cfg_path, _cfg = one_config(tmp_path, tmp_path / "bc.pt")
    args = [a.format(tmp=tmp_path, cfg=cfg_path) for a in command]
    result = run(args[0], "-c", cfg_path, *args[1:], "--set", "networks.model_class=no_such_module.Model")
    assert result.exit_code == 1 and "no_such_module" in result.output, result.output
