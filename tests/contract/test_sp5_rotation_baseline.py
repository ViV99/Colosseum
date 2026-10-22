"""Owner rotation and lineups on today's code (SP5 T0.1, spec §6 "Контрактные тесты до переноса", block 4.2): the
sequence of owners the launcher produces over several refreshes for 1-3 workers, 1-3 envs per worker and 1-3
trainable agents; every lineup's owner collects; the launcher's refresh advances one round and uses global env
offsets. The expectation is ``game_helpers.owner_sequence``; T2.1 runs it against ``hub.state.OwnerRotation`` (one
node, every cell expiring at once); T2.5 ports the launcher test to the hub (same owner sequence): the deficit rule
through ``Hub._on_lineup_request`` with every cell of one node asking at once (spec 4.2)."""
from __future__ import annotations

import pytest

from colosseum.net.messages import lineup_from_proto
from game_helpers import make_coordinator, make_test_config, owner_sequence
from hub_kit import accept_version, lineup_hub, open_node, request_cells

ROUNDS = 5
CASES = [(workers, envs, agents) for workers in (1, 2, 3) for envs in (1, 2, 3) for agents in (1, 2, 3)]


def _config(workers: int, envs: int, agents: int):
    ids = ["a", "b", "c"][:agents]
    return make_test_config("turns", agents={aid: {} for aid in ids},
                            rollout={"num_workers": workers, "envs_per_worker": envs},
                            matchmaking={"opponents": {"latest": 1.0, "snapshots": 0.0, "rivals": 0.0, "anchors": 0.0}})


def _owner(lineup) -> str:
    collecting = {s.agent_id for s in lineup.seats if s.collect and s.source in ("owner", "")}
    owners = {s.agent_id for s in lineup.seats if s.source == "owner"}
    assert owners and owners <= collecting, lineup
    return owners.pop()


@pytest.mark.parametrize("workers, envs, agents", CASES)
def test_the_owner_sequence_over_refreshes(tmp_path, workers, envs, agents):
    """The launcher's order: initial lineups at round 0 (``_start_children``), then per refresh ``next_round`` and one
    ``generate_lineups(E, w * E)`` per worker (``_refresh_worker_matches``)."""
    coord = make_coordinator(_config(workers, envs, agents), tmp_path / "ckpt")
    seen = []
    for r in range(ROUNDS):
        if r:
            coord.next_round()
        seen.append([[_owner(lu) for lu in coord.generate_lineups(envs, env_offset=w * envs)] for w in range(workers)])
    assert seen == owner_sequence(workers, envs, agents, ROUNDS)
    flat = [o for rnd in seen for worker in rnd for o in worker]
    counts = {aid: flat.count(aid) for aid in set(flat)}
    if (workers * envs) % agents == 0:
        assert len(set(counts.values())) == 1                     # equal shares when the cells divide evenly


def test_the_hub_rotates_one_round_per_refresh_over_global_env_offsets(tmp_path):
    """SP4's refresh (``next_round`` + per-worker lineups at global env offsets) on the hub: one node with 3 workers x 2
    envs whose cells all ask again at once (their TTL expired) gets round ``r`` of SP4's owner sequence."""
    hub = lineup_hub(_config(3, 2, 3), tmp_path)
    node = open_node(hub, workers=3)
    for agent in ("a", "b", "c"):
        assert accept_version(hub, agent) == "accepted"
    expected = owner_sequence(3, 2, 3, rounds=3)
    cells = {(w, e): 0 for w in range(3) for e in range(2)}           # epoch of every cell (0: none yet)
    for r in range(3):
        grants = request_cells(hub, node, [(w, e, epoch, 0) for (w, e), epoch in sorted(cells.items())])
        owners = [[None] * 2 for _ in range(3)]
        for grant in grants:
            owners[grant.worker][grant.env] = _owner(lineup_from_proto(grant.lineup))
            cells[(grant.worker, grant.env)] = grant.epoch
        assert owners == expected[r]
