"""Where a version is fetched from (SP5 T3.1, spec 4.5 "где взять", 3.1): the hub announces every accepted version
with the announcing node's number and data addresses; a node without a data address is relay-only."""
from __future__ import annotations

import dataclasses

from colosseum.hub.state import VersionInfo, version_source
from hub_kit import make_node_entry


def test_announced_versions_carry_the_announcing_nodes_addresses():
    info = VersionInfo(agent_id="main", generation=1, version=4, size=9, digest="d", node=99, addresses=("6.6.6.6:1",),
                       teacher_active=True)
    entry = make_node_entry(number=2, data_addresses=("10.0.0.2:47102",))
    sourced = version_source(info, entry)
    assert (sourced.node, sourced.addresses, sourced.relay_only) == (2, ("10.0.0.2:47102",), False)
    assert dataclasses.replace(sourced, node=99, addresses=("6.6.6.6:1",)) == info


def test_a_node_without_a_data_address_is_relay_only():
    info = VersionInfo(agent_id="main", generation=1, version=4, size=9, digest="d", node=2)
    sourced = version_source(info, make_node_entry(number=2, data_addresses=()))
    assert sourced.relay_only is True and sourced.addresses == ()
