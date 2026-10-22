"""Latest weights of another machine's learner on a node (SP5 T3.1, spec 4.5, 4.6, 4.2; criterion 4): one download
per version whatever the number of workers, ``GetLatest`` polls without the hub, ``teacher_active`` carried to the
remote cache, a failed first fetch holds the lineup and counts ``lineup_fetch_failed``, a full cache is an ERROR
with the hint.

The node under test runs in a thread against ``node_kit.FakeHub`` (insecure); the learner's node is emulated by the
test: a weight cache of its own and a real ``DataServer`` (plain + the token secret) that serves it as node 7."""
from __future__ import annotations

import logging

import pytest

from cluster_kit import cluster_config
from colosseum.core.registry import build_model
from colosseum.core.types import Lineup, SeatAssignment
from colosseum.net.chunks import WeightsMeta, encode_weights
from colosseum.net.messages import VersionInfo, version_to_proto
from colosseum.node import weights
from colosseum.node.dataport import DataServer, frame_digest
from colosseum.node.weights import WeightCacheReader, WeightCacheWriter, shm_dir
from fault_proxy import TcpFaultProxy
from game_helpers import agent_role_of
from node_kit import FakeHub, run_node_in_thread, toy_node_options, wait_for

LINEUP = Lineup("2p", [SeatAssignment("main", "latest", False), SeatAssignment("random", "fixed", False)])
LEARNER_NODE = 7
SYNC_SEC = 0.5


def _config(tmp_path, sync_sec=SYNC_SEC):
    return cluster_config(tmp_path, {"agents.main": "{}",
                                     "agents.random": "{kind: scripted, class: colosseum.players.RandomBot}",
                                     "rollout.envs_per_worker": 2,
                                     "rollout.weight_sync_interval_sec": sync_sec})


class LearnerNode:
    """The learner's machine as the node under test sees it: frames of the tic-tac-toe model in a cache of its own,
    served by a real data port (plain + the token secret)."""

    def __init__(self, tmp_path, config, token_secret):
        _roles, role = agent_role_of(config, "main")
        model = build_model(config.get_agent_config("main"), role)
        self.state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        self.writer = WeightCacheWriter(tmp_path / "learner-shm", "L")
        self.server = DataServer(cache=WeightCacheReader(tmp_path / "learner-shm", "L"), own_agents=lambda: {"main"},
                                 max_message_mb=64, insecure=True, token_secret=token_secret, node=LEARNER_NODE)
        self.port = self.server.start(0, credentials=None, host="127.0.0.1")
        self.frames: dict[int, bytes] = {}

    def publish(self, version, *, teacher_active=False):
        frame = encode_weights(self.state, WeightsMeta(agent_id="main", kind="latest", generation=1,
                                                       policy_version=version, teacher_active=teacher_active))
        self.writer.publish_latest("main", frame, version=version, generation=1, teacher_active=teacher_active)
        self.frames[version] = frame

    def info(self, version, *, address=None, teacher_active=False, digest=None):
        return version_to_proto(VersionInfo(
            agent_id="main", generation=1, version=version, size=len(self.frames[version]),
            digest=digest or frame_digest(self.frames[version]), node=LEARNER_NODE,
            addresses=(address or f"127.0.0.1:{self.port}",), teacher_active=teacher_active))


@pytest.fixture
def setup(tmp_path):
    started = []

    def make(*, workers=1, grant=True, address=None, first_teacher_active=False, sync_sec=SYNC_SEC, digest=None):
        config = _config(tmp_path, sync_sec)
        hub = FakeHub(tmp_path, config, ttl_sec=1.0)
        learner = LearnerNode(tmp_path, config, hub.token.secret)
        learner.publish(1, teacher_active=first_teacher_active)
        hub.welcome(versions=[learner.info(1, address=address, teacher_active=first_teacher_active,
                                           digest=digest)])
        if grant:
            hub.grant(lambda worker, env: LINEUP)
        node = run_node_in_thread(toy_node_options(tmp_path, hub, workers=workers))
        started.append((hub, learner, node))
        return hub, learner, node

    yield make
    for hub, learner, node in started:
        if node.thread.is_alive():
            node.stop()
        hub.stop()
        learner.server.stop()


def _played(hub, version, worker=None):
    """Results whose ``main`` seat ended on ``version`` (of worker ``worker``: its uid's ``w<worker>_``)."""
    return [r for r in hub.results()
            if any(s.agent_id == "main" and s.version_end == version for s in r.seats)
            and (worker is None or f"/w{worker}_" in r.uid)]


def _node_cache(node):
    return WeightCacheReader(shm_dir(), node.agent.cache_prefix)


def test_one_download_per_version_whatever_the_number_of_workers(setup):
    hub, learner, node = setup(workers=2)                       # the two workers are the only loaded processes
    wait_for(lambda: _played(hub, 1), timeout=60)
    for v in range(2, 6):
        learner.publish(v)
        hub.announce(versions=[learner.info(v)])
        wait_for(lambda v=v: _played(hub, v), timeout=3 * SYNC_SEC + 10)
    wait_for(lambda: _played(hub, 5, worker=0) and _played(hub, 5, worker=1), timeout=3 * SYNC_SEC + 10)
    history = node.agent.fetch_history()["main"]
    assert history == sorted(set(history)) and history[-1] == 5   # strictly increasing: no version twice
    assert learner.server.counters["latest_served"] == len(history)
    beat = wait_for(lambda: [h for h in hub.heartbeats() if h.counters["weight_fetches"] == len(history)])[-1]
    assert beat.counters["weight_fetch_redundant"] == 0 and beat.counters["weight_fetch_bytes"] > 0
    assert node.stop() == 0


def test_a_zero_sync_interval_fetches_at_the_pump_cadence(setup):
    """``rollout.weight_sync_interval_sec: 0`` ("pull every step", valid since SP1) runs: the node checks versions at
    its fetch pump's cadence (review round 1: it used to fail the run's start)."""
    hub, learner, node = setup(workers=1, sync_sec=0)
    wait_for(lambda: _played(hub, 1), timeout=60)
    learner.publish(2)
    hub.announce(versions=[learner.info(2)])
    wait_for(lambda: _played(hub, 2), timeout=30)
    assert node.agent.fetch_history() == {"main": [1, 2]}
    assert node.stop() == 0


def test_without_the_hub_versions_come_from_get_latest(setup):
    hub, learner, node = setup(workers=1)
    wait_for(lambda: _played(hub, 1), timeout=60)
    hub.pause()                                                 # the hub goes away (sessions refused)
    wait_for(lambda: learner.server.counters["latest_not_modified"] >= 1, timeout=3 * SYNC_SEC + 10)
    learner.publish(2)                                          # no announcement
    wait_for(lambda: _node_cache(node).latest_version("main") == (2, 1), timeout=3 * SYNC_SEC + 5)
    assert node.agent.counters()["weight_fetches"] == 2
    hub.resume()
    assert node.stop() == 0


def test_teacher_active_off_reaches_the_remote_cache(setup):
    hub, learner, node = setup(workers=1, first_teacher_active=True)
    wait_for(lambda: _node_cache(node).latest_version("main") == (1, 1), timeout=60)
    assert _node_cache(node).read_latest("main").meta.teacher_active is True
    learner.publish(2, teacher_active=False)
    hub.announce(versions=[learner.info(2)])
    wait_for(lambda: _node_cache(node).latest_version("main") == (2, 1), timeout=3 * SYNC_SEC + 10)
    cache = _node_cache(node)
    assert cache.read_latest("main").meta.teacher_active is False
    f, pointer = cache.open_latest("main")
    with f:
        assert pointer["teacher_active"] is False and f.read() == learner.frames[2]   # byte for byte
    assert node.stop() == 0


def test_a_failed_first_fetch_holds_the_lineup_and_counts_lineup_fetch_failed(tmp_path, setup):
    with TcpFaultProxy() as proxy:
        proxy.drop()
        hub, learner, node = setup(workers=1, address=proxy.address)
        proxy.set_target(("127.0.0.1", learner.port))
        wait_for(lambda: any(h.counters["lineup_fetch_failed"] >= 1 for h in hub.heartbeats()), timeout=60)
        assert not hub.results() and node.agent.fetch_history() == {}
        assert node.agent.counters()["weight_fetch_failed"] >= 1
        proxy.restore()
        wait_for(lambda: _played(hub, 1), timeout=60)
        assert node.agent.fetch_history() == {"main": [1]}
        assert node.stop() == 0


def test_a_full_cache_is_an_error_with_the_hint_and_waits_for_space(setup, monkeypatch, caplog):
    real = weights.shutil.disk_usage
    node_root = shm_dir()                              # the node's cache root; the learner's cache is elsewhere
    full = {"on": True}

    def disk_usage(path):                              # the cache's free-space probe: 1 MiB free in the node's root
        usage = real(path)
        return usage._replace(free=2**20) if full["on"] and str(path) == str(node_root) else usage

    monkeypatch.setattr(weights.shutil, "disk_usage", disk_usage)
    with caplog.at_level(logging.WARNING):
        hub, learner, node = setup(workers=0, grant=False)
        wait_for(lambda: node.agent.counters().get("weight_fetch_failed", 0) >= 1, timeout=30)
        beats = len(hub.heartbeats())
        wait_for(lambda: len(hub.heartbeats()) >= beats + 8, timeout=30)      # about four sync intervals
        assert learner.server.counters["latest_served"] == 1                  # not downloaded again while full
        assert node.agent.counters()["weight_fetch_failed"] == 1 and node.agent.fetch_history() == {}
        full["on"] = False                                                      # space again
        wait_for(lambda: _node_cache(node).latest_version("main") == (1, 1), timeout=30)
        assert node.agent.fetch_history() == {"main": [1]} and learner.server.counters["latest_served"] == 2
        assert node.stop() == 0
    errors = [r for r in caplog.records if r.levelno == logging.ERROR and "--shm-size" in r.getMessage()]
    assert len(errors) == 1 and "emptyDir" in errors[0].getMessage()


def test_a_frame_that_is_not_the_announced_version_is_refused(setup):
    """The hub's digest is the anchor of a direct fetch: other bytes under the announced version are refused and
    counted; the corrected announcement is fetched (review round 1)."""
    hub, learner, node = setup(workers=1, digest="0" * 32)
    wait_for(lambda: node.agent.counters().get("weight_fetch_failed", 0) >= 1, timeout=30)
    assert node.agent.fetch_history() == {} and _node_cache(node).latest_version("main") is None
    hub.announce(versions=[learner.info(1)])
    wait_for(lambda: _played(hub, 1), timeout=60)
    assert node.agent.fetch_history() == {"main": [1]}
    assert node.stop() == 0


def test_the_local_node_without_a_certificate_has_no_data_port_and_says_so_at_info(tmp_path, caplog):
    """Ruling R30: ``train``'s local node without minted credentials (``train`` without ``--listen``) runs without a
    data port; no remote node needs one then: an INFO, not a WARNING."""
    from colosseum.node.agent import NodeOptions

    hub = FakeHub(tmp_path, _config(tmp_path))
    record, secret = hub.identity.nodes.add_ephemeral("local", "machine-local")
    options = NodeOptions(hub_address=hub.address, token=hub.token, loopback_plain=True, local=True, workers=0,
                          credentials=(record.node_id, 0, secret), data_port=0, log_dir=tmp_path / "local-logs")
    node = None
    try:
        with caplog.at_level(logging.INFO, logger="colosseum.node"):
            node = run_node_in_thread(options)
            wait_for(lambda: node.agent.run_id is not None, timeout=30)
            assert node.stop() == 0
        said = [r for r in caplog.records if "no data port" in r.getMessage()]
        assert [r.levelno for r in said] == [logging.INFO] and node.agent.data_port() is None
    finally:
        if node is not None and node.thread.is_alive():
            node.stop()
        hub.stop()


def test_the_local_node_with_minted_credentials_serves_mutual_tls_on_its_listen_host(tmp_path):
    """Ruling R30 (T3.2): ``train --listen``'s local node gets a certificate the hub mints in memory; its data port
    binds on the listen host, announces only that address, and accepts a node of the same CA that names node 0."""
    from colosseum.net.security import mint_node_credentials
    from colosseum.node.agent import NodeOptions
    from colosseum.node.dataport import fetch_latest, probe_direct
    from colosseum.node.fetch import NodeFetcher
    from net_kit import node_identity, register

    hub = FakeHub(tmp_path, _config(tmp_path))
    record, secret = hub.identity.nodes.add_ephemeral("local", "machine-local")
    minted = mint_node_credentials(hub.identity, record.node_id)
    options = NodeOptions(hub_address=hub.address, token=hub.token, loopback_plain=True, local=True, workers=0,
                          credentials=(record.node_id, 0, secret), data_port=0, data_host="127.0.0.1",
                          data_credentials=(minted.key_pem, minted.cert_pem, minted.ca_pem),
                          log_dir=tmp_path / "local-logs")
    node = None
    try:
        node = run_node_in_thread(options)
        wait_for(lambda: node.agent.data_port() is not None, timeout=30)
        port = node.agent.data_port()
        wait_for(lambda: hub.messages("addresses"), timeout=10)
        assert list(hub.messages("addresses")[-1].data_addresses) == [f"127.0.0.1:{port}"]
        client = node_identity(tmp_path, hub.identity.fingerprint, "remote")
        register(hub.identity, client, name="remote", machine_id="machine-remote")
        link = NodeFetcher.link(insecure=False, token_secret="", identity=client, max_message_mb=64,
                                connect_timeout_sec=5.0)
        found = probe_direct([f"127.0.0.1:{port}"], lambda a: link.make_channel(a, 0), 5.0)
        assert found is not None                                       # mutual TLS with the minted certificate
        with pytest.raises(Exception, match="NOT_FOUND"):              # reached (no learner of main runs there)
            fetch_latest(found[1], "main", -1, max_message_mb=64, timeout=10)
        found[1].close()
        assert node.stop() == 0
    finally:
        if node is not None and node.thread.is_alive():
            node.stop()
        hub.stop()
