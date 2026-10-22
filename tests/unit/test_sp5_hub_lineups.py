"""The hub's lineups (SP5 T2.5, T3.2; spec 4.2, 4.7; pre-flight rulings P-PF2-2, P-PF2-3, P-PF2-17): the
real ``Hub._on_lineup_request`` on its coordinator, cell table and rotation, driven in-process through
``hub_kit``'s recording session handles."""
from __future__ import annotations

import logging

from colosseum.league.lineups import check_lineup
from colosseum.net.messages import lineup_from_proto
from colosseum.net.proto import colosseum_v1_pb2 as pb
from game_helpers import make_test_config, owner_sequence, scripted_agent
from hub_kit import accept_version, lineup_hub, node_message, open_node, request_cells

LATEST_ONLY = {"opponents": {"latest": 1.0, "snapshots": 0.0, "rivals": 0.0, "anchors": 0.0}}


def _config(agents=("a", "b", "c"), *, matchmaking=None, rollout=None, **sections):
    return make_test_config("turns", agents={aid: {} for aid in agents} if isinstance(agents, tuple) else agents,
                            matchmaking=matchmaking or LATEST_ONLY, rollout=rollout or {"envs_per_worker": 4},
                            **sections)


def _owner(lineup) -> str:
    owners = {s.agent_id for s in lineup.seats if s.source == "owner"}
    assert len(owners) == 1, lineup
    return owners.pop()


def _collecting(lineup) -> set[str]:
    return {s.agent_id for s in lineup.seats if s.collect}


def test_the_hub_rotates_owners_like_sp4_on_one_node(tmp_path):
    hub = lineup_hub(_config(), tmp_path)
    node = open_node(hub, workers=2)
    for agent in ("a", "b", "c"):
        assert accept_version(hub, agent) == "accepted"
    expected = owner_sequence(2, 4, 3, rounds=3)
    cells = {(w, e): (0, 0) for w in range(2) for e in range(4)}        # (epoch, incarnation)
    for rnd in range(3):
        grants = request_cells(hub, node, [(w, e, ep, inc) for (w, e), (ep, inc) in sorted(cells.items())])
        assert len(grants) == 8
        for grant in grants:
            lineup = lineup_from_proto(grant.lineup)
            owner = _owner(lineup)
            assert _collecting(lineup) == {owner}
            assert owner == expected[rnd][grant.worker][grant.env], (rnd, grant.worker, grant.env)
            cells[(grant.worker, grant.env)] = (grant.epoch, 0)
    assert len({g.epoch for g in node.grants()}) == 8                     # the cells kept their epochs


def test_lineups_come_from_the_coordinators_matchmaker(tmp_path):
    agents = {"a": {}, "rnd": scripted_agent()}
    mm = {"opponents": {"latest": 0.5, "snapshots": 0.0, "rivals": 0.0, "anchors": 0.5}}
    hub = lineup_hub(_config(agents, matchmaking=mm, rollout={"envs_per_worker": 4}), tmp_path)
    node = open_node(hub, workers=50)
    accept_version(hub, "a")
    grants = request_cells(hub, node, [(w, e, 0, 0) for w in range(50) for e in range(4)])
    assert len(grants) == 200
    sources = set()
    for grant in grants:
        lineup = lineup_from_proto(grant.lineup)
        check_lineup(hub._coordinator.context, lineup, "test")
        sources |= {s.source for s in lineup.seats if s.source != "owner"}
    assert {"latest", "anchors"} <= sources

    shuffled = lineup_hub(_config(agents, matchmaking={**mm, "shuffle_seats": True}), tmp_path / "s")
    node = open_node(shuffled, workers=50)
    accept_version(shuffled, "a")
    orders = {tuple(s.source == "owner" for s in lineup_from_proto(g.lineup).seats)
              for g in request_cells(shuffled, node, [(w, e, 0, 0) for w in range(50) for e in range(4)])}
    assert orders == {(True, False), (False, True)}


def test_every_grant_carries_its_epoch_and_the_refresh_ttl(tmp_path):
    hub = lineup_hub(_config(("a",), rollout={"envs_per_worker": 4, "match_refresh_interval_sec": 2.5}), tmp_path)
    node = open_node(hub, workers=1)
    accept_version(hub, "a")
    grants = request_cells(hub, node, [(0, e, 0, 0) for e in range(4)])
    for grant in grants:
        cell = hub._cells.get(next(k for k in (c.key for c in hub._cells.live()) if k.env == grant.env))
        assert grant.epoch == cell.epoch and lineup_from_proto(grant.lineup).epoch == cell.epoch
        assert grant.ttl_sec == 2.5
    assert sorted(g.epoch for g in grants) == [1, 2, 3, 4]


def test_a_refresh_of_zero_grants_each_cell_once(tmp_path):
    hub = lineup_hub(_config(("a",), rollout={"envs_per_worker": 2, "match_refresh_interval_sec": 0.0}), tmp_path)
    node = open_node(hub, workers=1)
    accept_version(hub, "a")
    grants = request_cells(hub, node, [(0, 0, 0, 0), (0, 1, 0, 0)])
    assert [g.ttl_sec for g in grants] == [0.0, 0.0]                      # 0: the node never asks again


def test_no_lineup_is_granted_while_no_agent_can_collect(tmp_path, caplog):
    """P-PF2-3 under R36: nothing can collect while no learner is placed (a node without learner capacity); a placed,
    alive learner is enough, before any first version."""
    hub = lineup_hub(_config(("a",)), tmp_path)
    workers_only = open_node(hub, name="worker-node", workers=1, cpu_learners=0, no_learners=True)
    assert hub._learners.current() == {} and hub._learners.to_place() == ["a"]
    with caplog.at_level(logging.DEBUG, logger="colosseum.hub.hub"):
        assert request_cells(hub, workers_only, [(0, e, 0, 0) for e in range(4)]) == []
    assert hub._cells.live() == [] and hub._epochs.reserved_until == 0    # P-PF2-3: no epoch allocated
    assert "no agent can collect yet" in caplog.text
    node = open_node(hub, name="learner-node", workers=1)                 # a's learner is placed: no version yet
    assert hub._versions.latest("a") is None
    grants = request_cells(hub, node, [(0, e, 0, 0) for e in range(4)])
    assert len(grants) == 4 and sorted(g.epoch for g in grants) == [1, 2, 3, 4]
    assert all(_collecting(lineup_from_proto(g.lineup)) == {"a"} for g in grants)


def test_agents_collect_from_their_learners_placement_without_a_first_version(tmp_path):
    """Ruling R36: with both learners placed and alive and only ``a``'s first version accepted, ``b``'s seats collect
    too (the node holds a lineup until ``b``'s latest weights are cached); once ``b``'s learner failed they do not, and
    (T4.1) ``b`` without any weights is not seated at all: with rivals only, no lineup is granted (counted). T4.2: a
    failed learner no longer fails the run (``a`` trains on), so the grants above are not an artefact of a failing
    hub."""
    mm = {"opponents": {"latest": 0.0, "snapshots": 0.0, "rivals": 1.0, "anchors": 0.0}}
    hub = lineup_hub(_config(("a", "b"), matchmaking=mm), tmp_path)
    node = open_node(hub, workers=2)
    accept_version(hub, "a")                                              # b's learner has published nothing
    assert hub._versions.latest("b") is None
    grants = request_cells(hub, node, [(w, e, 0, 0) for w in range(2) for e in range(4)])
    assert len(grants) == 8
    assert {_owner(lineup_from_proto(g.lineup)) for g in grants} == {"a", "b"}
    assert all(_collecting(lineup_from_proto(g.lineup)) == {"a", "b"} for g in grants)

    hub._on_message(node_message(node, event=pb.Event(kind="learner_failed", proc="learner-b", exit_code=1,
                                                      generation=1)))
    assert (hub._learners["a"].status, hub._learners["b"].status) == ("starting", "failed")
    assert hub._failure is None
    assert request_cells(hub, node, [(g.worker, g.env, g.epoch, 0) for g in grants]) == []
    assert hub._book.totals()["lineup_redraw_failed"] == 8


def test_a_node_without_a_path_to_the_learner_plays_without_collecting(tmp_path):
    """T3.2 (spec 4.7, 4.8): with ``distributed.relay: false`` a remote learner's seats collect only on a node that
    reported a direct path to it (``test_sp5_hub_paths.py``); before that the node plays the learner's newest stored
    snapshot, none collecting (R37)."""
    hub = lineup_hub(_config(("a",), distributed={"relay": False}), tmp_path)
    first = open_node(hub, name="learner-node", workers=1)
    second = open_node(hub, name="worker-node", workers=1, cpu_learners=0, no_learners=True)
    assert hub._learner_node_of("a") == first.of("welcome")[0].node_number
    assert [la.agent_id for la in first.of("welcome")[0].learners] == ["a"] and not second.of("welcome")[0].learners
    accept_version(hub, "a")
    hub._newest_snapshot = lambda agent: "ckpt_v4"                        # R37: snapshot seats there
    elsewhere = request_cells(hub, second, [(0, e, 0, 0) for e in range(4)])
    assert len(elsewhere) == 4                                            # granted, but nothing collects
    assert all(not _collecting(lineup_from_proto(g.lineup)) for g in elsewhere)
    here = request_cells(hub, first, [(0, e, 0, 0) for e in range(4)])
    assert all(_collecting(lineup_from_proto(g.lineup)) == {"a"} for g in here)


def test_cells_outside_the_nodes_slots_are_dropped_and_counted(tmp_path, caplog):
    hub = lineup_hub(_config(("a",)), tmp_path)
    node = open_node(hub, workers=1)
    accept_version(hub, "a")
    with caplog.at_level(logging.WARNING, logger="colosseum.hub.hub"):
        grants = request_cells(hub, node, [(0, 0, 0, 0), (5, 0, 0, 0), (0, 99, 0, 0), (-1, 0, 0, 0)])
        request_cells(hub, node, [(7, 7, 0, 0)])
    assert [(g.worker, g.env) for g in grants] == [(0, 0)]
    assert hub._book.get("node_messages.invalid") == 4
    assert caplog.text.count("node_messages.invalid") == 1                # one WARNING per node and reason
