"""Slow learning checks of SP2 (spec section 3, criterion 3): each demo game is trained with
``colosseum train`` on its example config (about 3 minutes or less on 8 cores, 2 workers), then
the newest checkpoint plays greedily against uniformly random legal players through the in-process
eval API (``play_lineups``). Thresholds are the spec's; changing one needs a ruling with
measurements (T8.3 Step 9), never a silent edit. ``test_tic_tac_toe_beats_random_80_percent`` is
the port of SP1's ``test_ttt_slow.py``."""
from __future__ import annotations

import json

import pytest

from colosseum.core.registry import env_spec
from colosseum.core.types import Lineup, SeatAssignment
from colosseum.eval import summarize
from demo_learning import (
    LEARNING_SEED,
    greedy_agent,
    mean_team_score,
    play,
    random_model,
    rotating_lineups,
    team_lineups,
    train_example,
    win_rate,
)
from examples.coop_buttons.game import measure_baselines

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1200), pytest.mark.usefixtures("restore_global_rng")]
TWO_WORKERS = {"rollout.num_workers": 2}


def _report(name: str, run, **numbers) -> None:
    text = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in numbers.items())
    print(f"[{name}] training {run.elapsed:.0f}s, {run.env_steps_per_sec():.0f} env steps/s; {text}")


def test_coin_grid_scores_at_least_twice_random(tmp_path):
    run = train_example("coin_grid", tmp_path, TWO_WORKERS)
    trained, role = greedy_agent(run, "agent_0")
    models = {"trained": trained, "random": random_model(role)}
    solo = [[Lineup("solo", [SeatAssignment(key)]) for _ in range(200)] for key in ("trained", "random")]
    score = mean_team_score(play(run.env_fn(), models, solo[0]), "trained")
    baseline = mean_team_score(play(run.env_fn(), models, solo[1]), "random")
    assert baseline > 0, f"random play scores {baseline}: the 2x criterion is meaningless"
    _report("coin_grid", run, score=score, random=baseline, ratio=score / baseline)
    assert score >= 2.0 * baseline


def test_tic_tac_toe_beats_random_80_percent(tmp_path):
    run = train_example("tic_tac_toe", tmp_path, TWO_WORKERS)
    trained, role = greedy_agent(run, "agent_0")
    (layout,) = env_spec(run.config).layouts
    results = play(run.env_fn(), {"trained": trained, "random": random_model(role)},
                   rotating_lineups(layout, 2, "trained", "random", 400))
    rate = win_rate(results, "trained")
    _report("tic_tac_toe", run, win_rate=rate)
    assert rate >= 0.80


def test_unit_harvest_beats_random_80_percent(tmp_path):
    run = train_example("unit_harvest", tmp_path, TWO_WORKERS)
    trained, role = greedy_agent(run, "agent_0")
    results = play(run.env_fn(), {"trained": trained, "random": random_model(role)},
                   rotating_lineups("2p", 2, "trained", "random", 200))
    rate = win_rate(results, "trained")
    _report("unit_harvest", run, win_rate=rate)
    assert rate >= 0.80


TEAM_TAG_RETRY_SEED_OFFSET = 1000


def _team_tag_win_rate(tmp_path, seed_offset: int) -> float:
    sets = {**TWO_WORKERS, "training.seed": LEARNING_SEED + seed_offset}
    run = train_example("team_tag", tmp_path, sets)
    trained, role = greedy_agent(run, "agent_0")
    spec = env_spec(run.config)
    results = play(run.env_fn(), {"trained": trained, "random": random_model(role)},
                   team_lineups(spec.teams("2v2"), "2v2", "trained", "random", 200))
    rate = win_rate(results, "trained")
    _report(f"team_tag seed+{seed_offset}", run, win_rate=rate)
    return rate


@pytest.mark.timeout(2400)
def test_team_tag_team_beats_random_team_80_percent(tmp_path):
    """Best of two independent runs (controller ruling, T8.3 fix round 2): about 2 of 9 single runs
    (the T8.3 record in the SP2 acceptance report) settle into a passive draw-seeking policy (0 losses,
    many draws), and async training is not reproducible per seed. If the first run misses the
    threshold, train once more from scratch (new run dir, seed offset ``TEAM_TAG_RETRY_SEED_OFFSET``).
    The threshold itself is the spec's. No other test retries."""
    first_dir = tmp_path / "run1"
    first_dir.mkdir()
    first = _team_tag_win_rate(first_dir, 0)
    if first >= 0.80:
        return
    retry_dir = tmp_path / "run2"
    retry_dir.mkdir()
    second = _team_tag_win_rate(retry_dir, TEAM_TAG_RETRY_SEED_OFFSET)
    print(f"[team_tag] best of two: first run {first:.3f}, retry {second:.3f}")
    assert second >= 0.80, f"both runs below 0.80: {first:.3f}, {second:.3f}"


def test_tron_wins_2p_and_takes_first_place_in_4p(tmp_path):
    run = train_example("tron", tmp_path, TWO_WORKERS)
    trained, role = greedy_agent(run, "agent_0")
    models = {"trained": trained, "random": random_model(role)}
    rate_2p = win_rate(play(run.env_fn(), models, rotating_lineups("2p", 2, "trained", "random", 200)), "trained")
    first_4p = win_rate(play(run.env_fn(), models, rotating_lineups("4p", 4, "trained", "random", 400)), "trained")
    _report("tron", run, win_rate_2p=rate_2p, first_place_4p=first_4p)
    assert rate_2p >= 0.80
    assert first_4p >= 0.50


def test_predator_prey_each_role_beats_random_70_percent(tmp_path):
    run = train_example("predator_prey", tmp_path, TWO_WORKERS)
    hunter, hunter_role = greedy_agent(run, "hunter")
    prey, prey_role = greedy_agent(run, "prey")
    models = {"hunter": hunter, "prey": prey,
              "random_hunter": random_model(hunter_role), "random_prey": random_model(prey_role)}

    def lineups(h: str, p: str) -> list[Lineup]:
        return [Lineup("1v2", [SeatAssignment(h), SeatAssignment(p), SeatAssignment(p)]) for _ in range(200)]

    hunter_rate = win_rate(play(run.env_fn(), models, lineups("hunter", "random_prey")), "hunter")
    prey_rate = win_rate(play(run.env_fn(), models, lineups("random_hunter", "prey")), "prey")
    random_rate = win_rate(play(run.env_fn(), models, lineups("random_hunter", "random_prey")), "random_hunter")
    _report("predator_prey", run, hunter_vs_random=hunter_rate, prey_vs_random=prey_rate,
            random_hunter_vs_random_prey=random_rate)
    assert hunter_rate >= 0.70
    assert prey_rate >= 0.70


def test_coop_buttons_homogeneous_teams_beat_the_measured_threshold(tmp_path):
    run = train_example("coop_buttons", tmp_path, TWO_WORKERS)
    a, role = greedy_agent(run, "coop_a")
    b, _ = greedy_agent(run, "coop_b")
    models = {"coop_a": a, "coop_b": b, "random": random_model(role)}

    def team(x: str, y: str, n: int = 100) -> list[Lineup]:
        return [Lineup("coop2", [SeatAssignment(x), SeatAssignment(y)]) for _ in range(n)]

    random_mean = mean_team_score(play(run.env_fn(), models, team("random", "random")), "random")
    oracle_mean = measure_baselines(episodes=300, **run.config.env.kwargs)["oracle"]
    threshold = max(5.0 * random_mean, 0.6 * oracle_mean)
    score_a = mean_team_score(play(run.env_fn(), models, team("coop_a", "coop_a")), "coop_a")
    score_b = mean_team_score(play(run.env_fn(), models, team("coop_b", "coop_b")), "coop_b")
    mixed = play(run.env_fn(), models, team("coop_a", "coop_b"))
    print(summarize(env_spec(run.config), mixed).text())
    cross = run.ratings()["layouts"]["coop2"]["cross_play"]
    _report("coop_buttons", run, coop_a=score_a, coop_b=score_b, threshold=threshold, random=random_mean,
            oracle=oracle_mean, cross_play=json.dumps(cross))
    assert oracle_mean > 0, "the oracle scores nothing: the threshold is meaningless"
    assert score_a >= threshold and score_b >= threshold
    assert cross.get("coop_a+coop_b", {}).get("n", 0) > 0          # the cross-play table was formed in training
