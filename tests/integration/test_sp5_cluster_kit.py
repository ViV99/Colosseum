"""``cluster_kit.hub_process`` (SP5 T3.1 test support, amendment A4): a ``colosseum hub`` CLI process on an ephemeral
loopback port whose token file names its identity; Ctrl+C stops it with 130."""
from __future__ import annotations

from cli_runner import TINY, TTT_CONFIG
from cluster_kit import hub_process
from colosseum.hub.registry import read_hub_json
from colosseum.net.security import parse_token


def test_hub_process_writes_its_token_file_and_stops_on_ctrl_c(tmp_path):
    hub = hub_process(TTT_CONFIG, tmp_path, "proc", overrides=dict(TINY))
    try:
        info = read_hub_json(hub.root)
        token = parse_token(hub.token_file.read_text())
        assert token.ca_fingerprint == info.ca_fingerprint and info.tls and info.mode == "hub"
        assert hub.address == info.addresses[0] and hub.address.startswith("127.0.0.1:")
        hub.interrupt()
        assert hub.wait(60) == 130, hub.tail()
    finally:
        if hub.proc.poll() is None:
            hub.kill()
