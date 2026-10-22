"""Hub-held ("parked") weights in the version registry (SP5 T4.1; spec 2.6 «соперником он играет последними
весами»; amendment A27): a parked version is the agent's latest until a learner version is accepted again or it is
unparked; it never lowers the version floor, its generation 0 is never stale, its node is -1 (node 0 is train's local
node), and it is not saved in hub_state.json (the hub parks again at start)."""
from __future__ import annotations

import json

import pytest

from colosseum.hub.state import HubStateData, VersionInfo, VersionRegistry, load_hub_state, save_hub_state

PARKED = VersionInfo("a", 0, 3, 100, "p3", -1, relay_only=True)


def _accepted(version: int, generation: int) -> VersionInfo:
    return VersionInfo("a", generation, version, 100, f"d{version}", 2)


def test_a_parked_version_is_latest_until_a_learner_version_is_accepted():
    reg = VersionRegistry()
    g = reg.new_generation("a")
    assert reg.announce(_accepted(5, g)) == "accepted"
    reg.park(PARKED)
    assert reg.latest("a") == PARKED and reg.parked("a") == PARKED
    assert reg.accepted("a").version == 5                  # the accepted latest stays known under the parked one
    assert reg.version_floor("a") == 6                       # parking never lowers the floor
    assert reg.is_stale("a", 0, 3) is False                   # generation 0 (hub-held) is never stale
    g2 = reg.new_generation("a")
    assert reg.announce(_accepted(6, g2)) == "accepted"
    assert reg.parked("a") is None and reg.latest("a").version == 6


def test_a_refused_announcement_keeps_the_parked_weights():
    reg = VersionRegistry()
    g = reg.new_generation("a")
    reg.announce(_accepted(5, g))
    reg.park(PARKED)
    assert reg.announce(_accepted(5, g)) == "not_newer"
    assert reg.parked("a") == PARKED


def test_unpark_restores_the_last_accepted_version():
    reg = VersionRegistry()
    reg.announce(_accepted(5, reg.new_generation("a")))
    reg.park(PARKED)
    assert reg.unpark("a") == PARKED
    assert reg.latest("a").version == 5 and reg.unpark("a") is None


def test_parked_weights_are_hub_held_and_not_saved(tmp_path):
    reg = VersionRegistry()
    with pytest.raises(ValueError, match="node -1"):
        reg.park(VersionInfo("a", 0, 3, 1, "x", 0))        # node 0 is train's local node (A27)
    with pytest.raises(ValueError, match="generation 0"):
        reg.park(VersionInfo("a", 4, 3, 1, "x", -1))
    reg.park(PARKED)
    again = VersionRegistry.from_state(json.loads(json.dumps(reg.state())))
    assert again.parked("a") is None and again.latest("a") is None
    data = HubStateData(version=1, run_id="r", env_steps=0, sessions={}, epoch_reserved_until=0,
                        versions=reg.state(), counters={}, nodes_lost=0, learners={"version": 1, "agents": {}})
    save_hub_state(tmp_path / "hub_state.json", data, fsync=False)
    assert load_hub_state(tmp_path / "hub_state.json").learners == {"version": 1, "agents": {}}
    raw = json.loads((tmp_path / "hub_state.json").read_text())
    del raw["learners"]                                        # a hub_state.json of T2.5 / T3.x
    (tmp_path / "hub_state.json").write_text(json.dumps(raw))
    assert load_hub_state(tmp_path / "hub_state.json").learners == {}
