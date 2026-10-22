"""Two ``train`` hubs on one machine do not collide (SP5 T2.5; amendment A14; criterion 6's contract part; the
two real ``colosseum train`` runs are T7.6's runbook step): two ``Hub`` objects in ``train`` mode in one process
(no local node: no loaded process), one hub identity, both serving a session at once."""
from __future__ import annotations

import threading
import time

from colosseum.hub.hub import Hub, HubOptions
from colosseum.hub.registry import RUNNING_DIR, read_hub_json, running_hubs
from colosseum.hub.server import LOOPBACK_KEY
from colosseum.net.local import address
from colosseum.node.weights import cache_prefix
from game_helpers import make_test_config, make_test_run_dir
from hub_kit import FakeNode


def _start(tmp_path, name):
    config = make_test_config("turns", training={"total_timesteps": 10**12})
    hub = Hub(config, make_test_run_dir(config, tmp_path, name=name),
              options=HubOptions(mode="train", local_node=None, identity_dir=tmp_path / "hub-identity"))
    codes: list[int] = []
    thread = threading.Thread(target=lambda: codes.append(hub.launch()), name=name, daemon=True)
    thread.start()
    deadline = time.monotonic() + 60
    while not hub.ports and time.monotonic() < deadline:
        time.sleep(0.05)
    assert hub.ports, f"{name} did not start"
    return hub, thread, codes


def test_two_train_hubs_on_one_machine_do_not_collide(tmp_path):
    started = [_start(tmp_path, "one"), _start(tmp_path, "two")]
    nodes = []
    try:
        (one, _, _), (two, _, _) = started
        ports = [h.ports[LOOPBACK_KEY] for h, _, _ in started]
        assert ports[0] != ports[1] and all(ports)
        infos = [read_hub_json(h._run_dir) for h, _, _ in started]
        assert {i.run_id for i in infos} == {one.run_id, two.run_id} and one.run_id != two.run_id
        assert [i.loopback for i in infos] == [f"127.0.0.1:{p}" for p in ports]
        assert {i.run_dir for i in running_hubs(one.identity.path)} == {str(one._run_dir.root), str(two._run_dir.root)}
        assert len(list((one.identity.path / RUNNING_DIR).glob("*.json"))) == 2
        fingerprint = one.identity.fingerprint
        assert cache_prefix(fingerprint, one.run_id, 0) != cache_prefix(fingerprint, two.run_id, 0)
        assert address(fingerprint[:8], one.run_id[:8], 0, "node") != address(fingerprint[:8], two.run_id[:8], 0,
                                                                               "node")
        for (hub, _, _), port in zip(started, ports, strict=True):     # both serve a session at once
            node = FakeNode(f"127.0.0.1:{port}", token=hub.identity.token, insecure=True)
            node.register(machine_id=f"m-{port}", name="probe")
            node.open("s1")
            nodes.append(node)
        welcomes = [n.wait_for("welcome")[0] for n in nodes]
        assert [w.run_id for w in welcomes] == [one.run_id, two.run_id]
    finally:
        for node in nodes:
            node.close()
        for hub, _thread, _ in started:
            hub.request_stop()
        for _, thread, _ in started:
            thread.join(60)
    assert [codes for _, _, codes in started] == [[0], [0]]
    assert not list((started[0][0].identity.path / RUNNING_DIR).glob("*.json"))
