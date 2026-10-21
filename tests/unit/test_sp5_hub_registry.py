"""hub.json and the running-hubs registry (SP5 T2.5; spec 1, 3.3)."""
from __future__ import annotations

import json
import multiprocessing as mp

from colosseum.hub.registry import (
    HUB_JSON,
    HubInfo,
    read_hub_json,
    register_running,
    running_hubs,
    this_process_info,
    unregister_running,
    write_hub_json,
)


def _info(**changes):
    base = dict(run_id="R", run_dir="/runs/x", addresses=("0.0.0.0:47100",), loopback="127.0.0.1:41234",
                tls=False, ca_fingerprint="f" * 64, identity_dir="/id", mode="train")
    return this_process_info(**{**base, **changes})


def test_hub_json_round_trips_and_a_malformed_one_reads_as_none(tmp_path):
    info = _info()
    write_hub_json(tmp_path, info)
    assert read_hub_json(tmp_path) == info and not info.loopback_tls
    assert _info(mode="hub", tls=True).loopback_tls
    (tmp_path / HUB_JSON).write_text("{")
    assert read_hub_json(tmp_path) is None and read_hub_json(tmp_path / "missing") is None


def _noop():
    pass


def test_the_running_registry_lists_live_hubs_and_removes_stale_entries(tmp_path):
    register_running(tmp_path, _info())
    proc = mp.get_context("spawn").Process(target=_noop)
    proc.start()
    proc.join()
    dead = HubInfo(**{**json.loads(_info().to_json()), "addresses": (), "pid": proc.pid, "proc_start": None})
    register_running(tmp_path, dead)
    (tmp_path / "running" / "999999999.json").write_text("not json")
    assert [h.pid for h in running_hubs(tmp_path)] == [_info().pid]
    assert sorted(p.name for p in (tmp_path / "running").iterdir()) == [f"{_info().pid}.json"]
    unregister_running(tmp_path, _info().pid)
    assert running_hubs(tmp_path) == []
