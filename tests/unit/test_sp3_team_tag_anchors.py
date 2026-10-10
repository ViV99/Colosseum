"""team_tag anchors (spec block 9, T6.3): the pre-fixed decision rule of scripts/team_tag_anchors.py
and the v3 shape of configs/examples/team_tag.yaml. No training here."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

from colosseum.core.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "scripts" / "team_tag_anchors.py"
_spec = importlib.util.spec_from_file_location("team_tag_anchors", _SCRIPT)
tta = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tta)


def _rows(wins: dict[float, list[float]], draw_reward=None) -> list[dict]:
    return [{"anchors": a, "draw_reward": draw_reward, "seed": s, "win": w}
            for a, ws in wins.items() for s, w in enumerate(ws)]


def test_the_highest_minimum_among_passing_shares_wins():
    rows = _rows({0.1: [0.9, 0.7, 0.9, 0.9, 0.9, 0.9], 0.2: [0.85, 0.82, 0.9, 0.9, 0.9, 0.9],
                  0.3: [0.88, 0.87, 0.9, 0.9, 0.9, 0.9]})
    choice, detail = tta.choose_anchor_share(rows)
    assert choice == 0.3
    assert any(line.startswith("anchors 0.1: min 0.700") for line in detail)


def test_a_smaller_share_within_the_tie_margin_wins():
    rows = _rows({0.2: [0.85] * 6, 0.3: [0.86] * 6})
    assert tta.choose_anchor_share(rows)[0] == 0.2


def test_no_passing_share_and_too_few_seeds_give_none():
    assert tta.choose_anchor_share(_rows({0.1: [0.9, 0.9, 0.9, 0.9, 0.9, 0.79]}))[0] is None
    assert tta.choose_anchor_share(_rows({0.1: [0.95] * 5}))[0] is None              # needs 6 seeds
    assert tta.choose_anchor_share(_rows({0.1: [0.95] * 6}, draw_reward=-0.5))[0] is None   # other draw reward


def test_the_draw_reward_fallback_prefers_the_mildest_passing_penalty():
    rows = ([{"anchors": 0.3, "draw_reward": d, "seed": s, "win": w} for s in range(6)
             for d, w in ((-0.25, 0.86), (-0.5, 0.87))]
            + [{"anchors": 0.1, "draw_reward": -0.25, "seed": s, "win": 0.99} for s in range(6)])
    assert tta.choose_draw_reward(rows, anchors=0.3)[0] == -0.25


def test_team_tag_example_is_v3_with_a_random_bot_anchor():
    path = REPO_ROOT / "configs" / "examples" / "team_tag.yaml"
    raw = yaml.safe_load(path.read_text())
    assert not set(raw["matchmaking"]) & {"mode", "self_play_ratio", "latest_prob", "pfsp_exponent"}
    assert "pool_size" not in raw["checkpoint"]
    config = load_config(path)
    assert config.get_trainable_agent_ids() == ["agent_0"]          # implicit, the agents section has only the bot
    assert config.fixed_agent_ids() == ["random"]
    assert config.agent_entry("random").class_path == "colosseum.players.RandomBot"
    assert config.matchmaking.anchors == ["random"]
    shares = config.matchmaking.opponents
    assert shares.anchors > 0 and shares.snapshots == 0.2 and shares.rivals == 0.0
    assert abs(shares.latest + shares.anchors - 0.8) < 1e-9
