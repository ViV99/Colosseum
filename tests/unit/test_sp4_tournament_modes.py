"""Tournament modes on a substituted results source (SP4 T5.3, spec block 3): gauntlet, round-robin, halving,
select, several layouts, role groups, the report schema."""
from __future__ import annotations

import json
import logging

import pytest

from colosseum.arena.tournament.modes import TournamentSettings, run
from colosseum.core.errors import ConfigError
from colosseum.envs.game import GameSpec, SeatSpec
from game_helpers import AsymmetricGame, CoopGame, FakeArena, TurnTakingGame, synthetic_players

LADDER = {"main@v1": -1.0, "main@v2": -0.5, "main@v3": 0.0, "main@v4": 0.5, "main@v5": 2.0,
          "greedy": 0.0, "random": -1.0}
CANDIDATES = [n for n in LADDER if "@v" in n]


def settings(**kw):
    base = dict(name="t", layouts=["2p"], layout_weights={"2p": 1.0}, blocks=40, halving_blocks=4,
                confirm_blocks=200, finalists=2, seed=3, bootstrap_samples=400)
    return TournamentSettings(**{**base, **kw})


def play(mode, spec, strengths, candidates, field, arena=None, players=None, roles=None, **kw):
    players = players or synthetic_players(strengths, roles)
    arena = arena or FakeArena(spec, players, strengths)
    return run(mode, spec=spec, players=players, candidates=candidates, field=field, settings=settings(**kw),
               play=arena).to_dict()


def test_select_halves_then_confirms_the_strongest_on_fresh_seeds():
    spec = TurnTakingGame().spec
    players = synthetic_players(LADDER)
    arena = FakeArena(spec, players, LADDER)
    report = play("select", spec, LADDER, CANDIDATES, ["greedy", "random"], arena=arena, players=players)
    rounds = report["rounds"]
    assert [r["blocks_per_composition"] for r in rounds] == [4, 8]
    assert [len(r["active"]) for r in rounds] == [5, 3] and [len(r["eliminated"]) for r in rounds] == [2, 1]
    sel = report["selection"]["player"]
    assert sel["winner"] == "main@v5" and sel["confirmed"] is True and sel["diff_ci95"][0] > 0
    assert "biased up" in sel["note"]
    stage_a = {i for call in arena.calls for i in call if "/A/" in i}
    stage_b = {i for call in arena.calls for i in call if "/B/" in i}
    assert stage_a and stage_b and not stage_a & stage_b
    tables = {(t["stage"], t["layout"]): t for t in report["tables"]}
    assert set(tables) == {("A", "2p"), ("B", "2p")}
    b_rows = tables[("B", "2p")]["rows"]
    assert b_rows[0]["name"] == "main@v5" and len(b_rows) == 2
    assert report["games"]["by_stage"]["B"]["blocks"] == 2 * 2 * 200
    json.dumps(report)


def test_select_does_not_confirm_indistinguishable_finalists():
    spec = TurnTakingGame().spec
    flat = {"main@v1": 0.0, "main@v2": 0.0, "main@v3": 0.0, "greedy": 0.0}
    players = synthetic_players(flat)
    arena = FakeArena(spec, players, flat, crn=True)    # common random numbers: identical finalists, identical games
    report = play("select", spec, flat, ["main@v1", "main@v2", "main@v3"], ["greedy"], confirm_blocks=60,
                  arena=arena, players=players)
    sel = report["selection"]["player"]
    assert sel["confirmed"] is False and sel["runner_up"] in sel["indistinguishable"]
    assert sel["diff"] == 0.0 and sel["diff_ci95"] == [0.0, 0.0]


def test_halving_accumulates_rounds_and_stops_at_the_finalists():
    spec = TurnTakingGame().spec
    report = play("halving", spec, LADDER, CANDIDATES, ["greedy"], finalists=1)
    assert [len(r["active"]) for r in report["rounds"]] == [5, 3, 2]
    assert [r["first_index"] for r in report["rounds"]] == [0, 4, 12]
    assert report["selection"] is None
    (table,) = [t for t in report["tables"] if t["stage"] == "A"]
    assert [r["name"] for r in table["rows"]] == ["main@v5"] and table["rows"][0]["blocks"] == 4 + 8 + 16
    with pytest.raises(ConfigError, match="mode"):
        play("swiss", spec, LADDER, CANDIDATES, ["greedy"])


def test_gauntlet_tables_worst_opponent_and_matrix():
    spec = TurnTakingGame().spec
    report = play("gauntlet", spec, LADDER, ["main@v5", "main@v1"], ["greedy", "random"], blocks=30)
    (table,) = report["tables"]
    top = table["rows"][0]
    assert top["name"] == "main@v5" and top["worst"]["opponent"] == "greedy"
    cell = table["matrix"]["main@v5"]["random"]
    assert (cell["blocks"], cell["games"]) == (30, 60) and len(cell["ci95"]) == 2
    assert top["games"] == 120 and top["blocks"] == 60


def test_several_layouts_are_combined_with_the_layout_weights():
    spec = GameSpec(roles=TurnTakingGame().spec.roles,
                    layouts={"2p": (SeatSpec("player", 0), SeatSpec("player", 1)),
                             "2v": (SeatSpec("player", 0), SeatSpec("player", 1))})
    report = play("gauntlet", spec, LADDER, ["main@v5", "main@v1"], ["greedy"], layouts=["2p", "2v"],
                  layout_weights={"2p": 3.0, "2v": 1.0}, blocks=20)
    (combined,) = [t for t in report["tables"] if t["layout"] == "*"]
    per = {t["layout"]: {r["name"]: r["score"] for r in t["rows"]} for t in report["tables"] if t["layout"] != "*"}
    got = {r["name"]: r["score"] for r in combined["rows"]}["main@v5"]
    assert got == pytest.approx(0.75 * per["2p"]["main@v5"] + 0.25 * per["2v"]["main@v5"])
    from colosseum.arena.tournament.modes import report_view

    view = report_view(report)                              # the combination reaches every layout's group view
    for layout in ("2p", "2v"):
        (group,) = view["layouts"][layout]["groups"].values()
        assert {r["name"]: r["score"] for r in group["combined"]}["main@v5"] == got
    with pytest.raises(ConfigError, match="weight"):
        play("gauntlet", spec, LADDER, ["main@v5"], ["greedy"], layouts=["2p", "2v"], layout_weights={"2p": 1.0})


def test_asymmetric_roles_select_one_winner_per_role_group():
    spec = AsymmetricGame().spec
    strengths = {"h@v1": 1.0, "h@v2": -1.0, "y@v1": 1.0, "y@v2": -1.0, "bot": 0.0}
    roles = {"h@v1": ("hunter",), "h@v2": ("hunter",), "y@v1": ("prey",), "y@v2": ("prey",), "bot": ("hunter", "prey")}
    report = play("select", spec, strengths, ["h@v1", "h@v2", "y@v1", "y@v2"], ["bot"], roles=roles,
                  layouts=["1v2"], layout_weights={"1v2": 1.0})
    assert {g: s["winner"] for g, s in report["selection"].items()} == {"hunter": "h@v1", "prey": "y@v1"}


def test_score_layouts_and_the_mixed_kind_rule():
    spec = CoopGame(size=2).spec
    strengths = {"main@v1": 1.0, "main@v2": 3.0, "helper": 0.0}
    report = play("select", spec, strengths, ["main@v1", "main@v2"], ["helper"], layouts=["coop2"],
                  layout_weights={"coop2": 1.0}, confirm_blocks=30)
    assert report["selection"]["player"]["winner"] == "main@v2" and report["bt"] == {"coop2": None}
    table = next(t for t in report["tables"] if t["stage"] == "B")
    assert table["metric"] == "score" and table["columns"] == ["(self)", "helper"]
    mixed = GameSpec(roles=spec.roles, layouts={"coop2": spec.layouts["coop2"],
                                                "2p": (SeatSpec("player", 0), SeatSpec("player", 1))})
    with pytest.raises(ConfigError, match="score"):
        play("select", mixed, strengths, ["main@v1", "main@v2"], ["helper"], layouts=["coop2", "2p"],
             layout_weights={"coop2": 1.0, "2p": 1.0})
    # gauntlet keeps both kinds per layout, but never averages them
    report = play("gauntlet", mixed, strengths, ["main@v1", "main@v2"], ["helper"], layouts=["coop2", "2p"],
                  layout_weights={"coop2": 1.0, "2p": 1.0}, blocks=5)
    assert sorted(t["layout"] for t in report["tables"]) == ["2p", "coop2"]


def test_round_robin_matrix_is_antisymmetric():
    spec = TurnTakingGame().spec
    report = play("round-robin", spec, LADDER, ["main@v5", "main@v1"], ["greedy"], blocks=20)
    (table,) = report["tables"]
    m = table["matrix"]
    assert m["main@v5"]["greedy"]["score"] == pytest.approx(1 - m["greedy"]["main@v5"]["score"])
    assert table["rows"][0]["name"] == "main@v5"


def test_the_report_text_names_tables_and_the_selection():
    spec = TurnTakingGame().spec
    players = synthetic_players(LADDER)
    rep = run("select", spec=spec, players=players, candidates=CANDIDATES, field=["greedy"], settings=settings(),
              play=FakeArena(spec, players, LADDER))
    text = rep.text()
    assert "stage B" in text and "winner main@v5 (confirmed" in text and "halving player round 0" in text


def test_the_report_view_by_layout_and_role_group():
    """Contract amendment A26: ``report_view`` gives Part 3's per-layout / per-group view of the report."""
    import functools

    from colosseum.arena.tournament.modes import bt_section, report_view

    spec = TurnTakingGame().spec
    players = synthetic_players(LADDER)
    data = run("select", spec=spec, players=players, candidates=CANDIDATES, field=["greedy"], settings=settings(),
               play=FakeArena(spec, players, LADDER),
               bt_fn=functools.partial(bt_section, players=players, samples=50)).to_dict()
    view = report_view(data)
    (group,) = view["layouts"]["2p"]["groups"].values()
    sel = data["selection"]["player"]
    assert view["winner"] == sel["winner"] == group["select"]["winner"] == group["ranking"][0]
    assert view["confirmed"] is sel["confirmed"] and sorted(group["ranking"]) == sorted(CANDIDATES)
    rows = {r["name"]: r for r in group["candidates"]}
    assert set(rows) == set(CANDIDATES) and all(r["bt_ci"] is not None for r in rows.values())
    out = [r for r in rows.values() if r["eliminated_round"] is not None]
    assert len(out) == len(CANDIDATES) - settings().finalists and all(r["score_ci"] is None for r in out)
    assert [r["blocks"] for r in group["rounds"]] == [r["blocks_per_composition"] for r in data["rounds"]]
    assert report_view({"name": "t", "mode": "gauntlet"}) == {"name": "t", "mode": "gauntlet", "seed": None,
                                                              "layouts": {}}
    assert report_view(view) == view                       # idempotent
    json.dumps(view)


def test_bt_section_rates_the_tournament_matches_with_block_intervals():
    """The tournament's own BT ratings (T2.1 library): order follows the true strengths on 120 blocks per
    composition (BT differences of 1.5 logits between neighbours: a reversal at this size is far below 1e-3)."""
    from colosseum.arena.tournament.modes import bt_section
    from colosseum.arena.tournament.schedule import gauntlet_blocks

    spec = TurnTakingGame().spec
    strengths = {"main@v1": -1.5, "main@v2": 0.0, "main@v3": 1.5, "greedy": 0.0}
    players = synthetic_players(strengths)
    blocks = gauntlet_blocks(spec, "2p", ["main@v1", "main@v2", "main@v3"], ["greedy"], players, blocks=120, seed=0,
                             stage="A")
    played = FakeArena(spec, players, strengths)(blocks)
    results = [r for b in blocks for r in played[b.id]]
    section = bt_section("2p", results, blocks, players=players, samples=50, seed=0)
    names = [e["name"] for e in section["entities"]]
    assert names.index("main@v3") < names.index("main@v2") < names.index("main@v1")
    for e in section["entities"]:
        assert e["ci95"][0] <= e["rating"] <= e["ci95"][1] and 0.0 <= e["p_best"] <= 1.0
    assert section["transitivity"]["flag"] is False and section["converged"] is True
    json.dumps(section)


def test_dropped_compositions_reach_the_report_and_the_view():
    """A composition no seat assignment fits (here: a hunter-only field member against a hunter candidate) is
    listed once per layout in ``to_dict()["layouts"]`` and in the view, also across halving rounds."""
    from colosseum.arena.tournament.modes import report_text, report_view

    spec = AsymmetricGame().spec
    strengths = {"h@v1": 1.0, "h@v2": -1.0, "h@v3": 0.0, "bot": 0.0, "hbot": 0.0}
    roles = {"h@v1": ("hunter",), "h@v2": ("hunter",), "h@v3": ("hunter",), "bot": ("hunter", "prey"),
             "hbot": ("hunter",)}
    report = play("select", spec, strengths, ["h@v1", "h@v2", "h@v3"], ["bot", "hbot"], roles=roles,
                  layouts=["1v2"], layout_weights={"1v2": 1.0}, confirm_blocks=20, finalists=1)
    assert len(report["rounds"]) == 2
    (entry,) = report["layouts"]
    assert sorted((d["candidate"], d["composition"]) for d in entry["dropped"]) == [
        ("h@v1", ["hbot"]), ("h@v2", ["hbot"]), ("h@v3", ["hbot"])]
    assert all("roles" in d["reason"] for d in entry["dropped"])
    view = report_view(report)
    assert view["layouts"]["1v2"]["dropped"] == entry["dropped"]
    assert "Not scheduled in 1v2: h@v1 vs hbot" in report_text(report)
    json.dumps(report, allow_nan=False)


def test_schedule_info_lines_are_logged_once_across_halving_rounds(caplog):
    """The builders log a dropped composition or a candidate that is also in the field on every call; a
    tournament logs each line once."""
    spec = TurnTakingGame().spec
    caplog.set_level(logging.INFO, logger="colosseum.arena.tournament.schedule")
    report = play("halving", spec, LADDER, ["main@v5", "main@v1", "main@v2", "main@v3", "greedy"],
                  ["greedy", "random"], finalists=1)
    assert len(report["rounds"]) == 3
    lines = [r.getMessage() for r in caplog.records if "also in the field" in r.getMessage()]
    assert lines == ["layout '2p': candidate 'greedy' is also in the field; it does not play itself"]
    assert report["flags"]["self_excluded"] == [["2p", "greedy"]]


def test_a_candidate_in_the_field_faces_the_field_without_itself():
    spec = TurnTakingGame().spec
    report = play("gauntlet", spec, LADDER, ["main@v5", "greedy"], ["greedy", "random"], blocks=10)
    (table,) = report["tables"]
    assert set(table["matrix"]["greedy"]) == {"random"} and set(table["matrix"]["main@v5"]) == {"greedy", "random"}
    assert report["flags"]["self_excluded"] == [["2p", "greedy"]]
    from colosseum.arena.tournament.modes import report_view

    assert report_view(report)["layouts"]["2p"]["self_excluded"] == ["greedy"]
    from colosseum.arena.tournament.modes import report_text

    assert "plays the field without itself" in report_text(report)


def test_repeated_pairs_are_flagged_in_a_deterministic_select():
    spec = TurnTakingGame().spec
    strengths = {"main@v1": 100.0, "main@v2": -100.0, "main@v3": -100.0, "greedy": 0.0}
    report = play("select", spec, strengths, ["main@v1", "main@v2", "main@v3"], ["greedy"], deterministic=True,
                  confirm_blocks=10)
    assert ["2p", "main@v1", "greedy"] in report["flags"]["repeated_pairs"]
    sel = report["selection"]["player"]
    assert sel["winner"] == "main@v1" and sel["confirmed"] is False and sel["repeated_columns"] == [["2p", "greedy"]]
    stage_b = next(t for t in report["tables"] if t["stage"] == "B")
    cell = stage_b["matrix"]["main@v1"]["greedy"]
    assert cell["repeated"] is True and cell["ci95"][0] < 0.25      # one block's Wilson interval, not a point
    json.dumps(report, allow_nan=False)
    from colosseum.arena.tournament.modes import report_view

    view = report_view(report)                              # the REPEATED marks reach the view (T7.1)
    (group,) = view["layouts"]["2p"]["groups"].values()
    rows = {r["name"]: r for r in group["candidates"]}
    assert "greedy" in rows["main@v1"]["repeated"] and group["matrix"]["main@v1"]["greedy"]["repeated"] is True
    assert group["select"]["repeated_columns"] == [["2p", "greedy"]]
    assert ["2p", "greedy"] in group["select"]["diff_repeated"] and group["combined"] is None
    assert report_view(view) == view


def test_halving_ranks_by_name_whatever_the_input_order():
    """Regression (T5.3 review): the strongest candidate comes last in the input; it survives every round, and
    every round's scores are the stage A scores over the blocks played so far."""
    from colosseum.arena.tournament.stats import candidate_scores

    spec = TurnTakingGame().spec
    ladder = {"main@v1": -3.0, "main@v2": -2.0, "main@v3": 0.0, "main@v4": 1.0, "main@v5": 3.0, "greedy": 0.0}
    players = synthetic_players(ladder)
    arena = FakeArena(spec, players, ladder)
    played: dict = {}
    seen: list = []

    def record(blocks):
        out = arena(blocks)
        played.update(out)
        seen.extend(blocks)
        return out

    names = {(p.agent_id, p.network_id): n for n, p in players.items()}
    for finalists in (1, 2):
        played.clear()
        seen.clear()
        report = run("halving", spec=spec, players=players, candidates=[n for n in ladder if "@v" in n],
                     field=["greedy"], settings=settings(halving_blocks=20, finalists=finalists), play=record).to_dict()
        assert all("main@v5" in r["active"] and "main@v5" not in r["eliminated"] for r in report["rounds"])
        assert report["rounds"][0]["eliminated"] == ["main@v2", "main@v1"]
        for r in report["rounds"]:
            upto = r["first_index"] + r["blocks_per_composition"]
            blocks = [b for b in seen if b.candidate in r["active"] and b.index < upto]
            (table,) = candidate_scores(blocks, played, names=names, lineage={n: p.lineage for n, p in players.items()},
                                        group_of={(b.layout, b.candidate): "player" for b in blocks},
                                        metric_of={"2p": "pairwise"}).values()
            expected = dict(zip(table.candidates, table.scores().tolist()))
            assert r["scores"] == pytest.approx(expected)
            assert r["active"] == sorted(expected, key=lambda n: -expected[n])


def test_repeated_score_pairs_claim_no_certainty_and_marks_follow_the_intervals():
    spec = CoopGame(size=2).spec
    strengths = {"main@v1": 1.0, "main@v2": 3.0, "helper": 0.0}
    players = synthetic_players(strengths)
    report = play("select", spec, strengths, ["main@v1", "main@v2"], ["helper"], layouts=["coop2"],
                  layout_weights={"coop2": 1.0}, confirm_blocks=5, deterministic=True, players=players,
                  arena=FakeArena(spec, players, strengths, noise=0.0))
    table = next(t for t in report["tables"] if t["stage"] == "B")
    cell = table["matrix"]["main@v2"]["(self)"]
    assert cell["repeated"] is True and cell["ci95"] is None
    assert table["rows"][0]["repeated"] == ["(self)", "helper"]
    sel = report["selection"]["player"]
    assert sel["confirmed"] is False and sel["diff_repeated"] == [["coop2", "(self)"], ["coop2", "helper"]]
    from colosseum.arena.tournament.modes import report_text

    assert "REPEATED games in coop2: (self)" in report_text(report)
    json.dumps(report, allow_nan=False)


def test_combined_rows_carry_the_repeated_columns():
    spec = GameSpec(roles=TurnTakingGame().spec.roles,
                    layouts={"2p": (SeatSpec("player", 0), SeatSpec("player", 1)),
                             "2v": (SeatSpec("player", 0), SeatSpec("player", 1))})
    strengths = {"main@v1": 100.0, "main@v2": -100.0, "greedy": 0.0}
    report = play("gauntlet", spec, strengths, ["main@v1", "main@v2"], ["greedy"], layouts=["2p", "2v"],
                  layout_weights={"2p": 1.0, "2v": 1.0}, blocks=4, deterministic=True)
    (combined,) = [t for t in report["tables"] if t["layout"] == "*"]
    assert {r["name"]: r["repeated"] for r in combined["rows"]} == {
        "main@v1": [["2p", "greedy"], ["2v", "greedy"]], "main@v2": [["2p", "greedy"], ["2v", "greedy"]]}


def test_a_self_excluded_candidate_reports_the_weights_it_uses():
    spec = TurnTakingGame().spec
    report = play("gauntlet", spec, LADDER, ["main@v5", "greedy"], ["greedy", "random", "main@v1"], blocks=4)
    (table,) = report["tables"]
    assert table["columns"] == ["greedy", "main@v1", "random"]
    assert table["column_weights"] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert set(table["candidate_weights"]) == {"greedy"}
    assert table["candidate_weights"]["greedy"] == pytest.approx([0.0, 0.5, 0.5])


def test_select_with_one_finalist_warns_that_nothing_can_be_confirmed(caplog):
    spec = TurnTakingGame().spec
    caplog.set_level(logging.WARNING, logger="colosseum.arena.tournament.modes")
    report = play("select", spec, LADDER, ["main@v5", "main@v1"], ["greedy"], finalists=1, confirm_blocks=5)
    assert report["selection"]["player"]["confirmed"] is False
    (line,) = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert "--finalists 1" in line and "colosseum pin" in line


def test_round_robin_marks_repeated_pairs_under_deterministic():
    """B-M1 (T5.3 deferred; spec block 3: repeated games count as one block): a deterministic round-robin pair whose
    blocks all repeat gets the Wilson interval of one observation and a REPEATED mark, both ways, and is listed once
    in ``flags.repeated_pairs``, as the gauntlet does; a pair with varying games and a stochastic run get none."""
    from colosseum.arena.tournament.modes import report_text
    from colosseum.eval import wilson_interval

    spec = TurnTakingGame().spec
    strengths = {"main@v1": 100.0, "main@v2": 0.0, "greedy": 0.0}
    report = play("round-robin", spec, strengths, ["main@v1", "main@v2"], ["greedy"], deterministic=True, blocks=10)
    (table,) = report["tables"]
    m = table["matrix"]
    lo, hi = wilson_interval(1.0, 1)
    assert m["main@v1"]["greedy"]["score"] == 1.0 and m["main@v1"]["greedy"]["repeated"] is True
    assert m["main@v1"]["greedy"]["ci95"] == pytest.approx([lo, hi])
    assert m["greedy"]["main@v1"]["repeated"] is True and m["greedy"]["main@v1"]["ci95"] == pytest.approx([1 - hi,
                                                                                                         1 - lo])
    assert m["main@v2"]["greedy"]["repeated"] is False and m["main@v2"]["greedy"]["ci95"][0] > lo   # games vary
    assert sorted(report["flags"]["repeated_pairs"]) == [["2p", "main@v1", "greedy"], ["2p", "main@v1", "main@v2"]]
    rows = {r["name"]: r for r in table["rows"]}
    assert rows["main@v1"]["repeated"] == ["greedy", "main@v2"] and rows["greedy"]["repeated"] == ["main@v1"]
    assert "Games repeat (not enough randomness): 2p: main@v1 vs greedy" in report_text(report)
    json.dumps(report, allow_nan=False)
    stochastic = play("round-robin", spec, strengths, ["main@v1", "main@v2"], ["greedy"], blocks=10)
    assert stochastic["flags"]["repeated_pairs"] == []
    assert not any(cell["repeated"] for cols in stochastic["tables"][0]["matrix"].values() for cell in cols.values())


def test_round_robin_refuses_mixed_role_groups_and_a_pairless_participant_has_no_score():
    """B-M8 (spec block 3: «Смешанные группы в одной таблице — ConfigError с перечнем групп»): hunters and prey are
    not ranked in one round-robin table; a participant without a playable pair gets ``score: null``, not 0."""
    from colosseum.arena.tournament import modes

    spec = AsymmetricGame().spec
    strengths = {"h@v1": 1.0, "y@v1": -1.0, "both": 0.0}
    roles = {"h@v1": ("hunter",), "y@v1": ("prey",), "both": ("hunter", "prey")}
    with pytest.raises(ConfigError, match=r"role groups.*'hunter': \['h@v1'\].*'prey': \['y@v1'\]"):
        play("round-robin", spec, strengths, ["h@v1"], ["y@v1"], roles=roles, layouts=["1v2"],
             layout_weights={"1v2": 1.0}, blocks=2)
    one_group = play("round-robin", spec, {"a@v1": 1.0, "b@v1": 0.0}, ["a@v1"], ["b@v1"], blocks=2, layouts=["1v2"],
                     layout_weights={"1v2": 1.0}, roles={"a@v1": ("hunter", "prey"), "b@v1": ("hunter", "prey")})
    assert {r["name"] for r in one_group["tables"][0]["rows"]} == {"a@v1", "b@v1"}
    players = synthetic_players(strengths, roles)
    ctx = modes._Context(spec, players, ["both"], ["h@v1"], settings(layouts=["1v2"], layout_weights={"1v2": 1.0},
                                                                         blocks=2), FakeArena(spec, players, strengths))
    ctx.run(ctx.round_robin("1v2", ["both", "h@v1"]))
    (table,) = modes._round_robin_tables(ctx, ["both", "h@v1", "y@v1"])      # y@v1 played no block
    rows = {r["name"]: r for r in table["rows"]}
    assert rows["y@v1"]["score"] is None and rows["y@v1"]["opponents"] == 0
    assert [r["name"] for r in table["rows"]][-1] == "y@v1"                  # unranked last
    report = {"name": "t", "mode": "round-robin", "seed": 0, "deterministic": False, "tables": [table], "rounds": [],
              "selection": None, "bt": {}, "flags": {"repeated_pairs": []}, "games": {"matches": 0, "blocks": 0}}
    assert "y@v1" in modes.report_text(report)


def test_round_robin_checks_the_role_groups_of_every_layout_before_playing_any():
    """FW-2 B-a: in layout ``aa`` both participants play role ``a`` (one group), in ``bc`` one plays ``b`` and the
    other ``c`` (two groups). The ConfigError comes before any block of ``aa`` is played: the results source,
    which in the CLI is the runner that journals every played block (``run.BlockRunner``), is never called."""
    role = TurnTakingGame().spec.roles["player"]
    spec = GameSpec(roles={"a": role, "b": role, "c": role},
                    layouts={"aa": (SeatSpec("a", 0), SeatSpec("a", 1)), "bc": (SeatSpec("b", 0), SeatSpec("c", 1))})
    strengths = {"p@v1": 0.0, "q@v1": 0.0}
    players = synthetic_players(strengths, {"p@v1": ("a", "b"), "q@v1": ("a", "c")})
    arena = FakeArena(spec, players, strengths)
    with pytest.raises(ConfigError, match=r"in layout 'bc' they play different role groups"):
        play("round-robin", spec, strengths, ["p@v1"], ["q@v1"], arena=arena, players=players, layouts=["aa", "bc"],
             layout_weights={"aa": 1.0, "bc": 1.0}, blocks=2)
    assert arena.calls == []
