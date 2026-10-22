"""The evicted-snapshot hold end to end (SP5 T3.1, spec 4.2): snapshots are evicted all the time while granted
lineups name them; every grant's snapshot can still be downloaded (from the store or ``checkpoints/.evicted/``), the
holds are counted in the hub's book, and nothing is left under ``.evicted`` after the stop.

One node in a thread with a CPU learner and one worker (the 2 loaded processes)."""
from __future__ import annotations

import time

from cluster_kit import cluster, cluster_config, hub_counter, journal_rows, wait_until
from colosseum.coordinator.checkpoint_manager import EVICTED_DIR
from colosseum.hub.server import BlobStore

DOWNLOAD_DELAY_SEC = 0.5      # each snapshot download starts this late: newer snapshots evict it meanwhile


def test_snapshots_evicted_while_referenced_stay_downloadable(tmp_path, monkeypatch):
    frame_path = BlobStore.frame_path

    def slow_frame_path(self, kind, agent_id, ident):          # the hub runs in this process (cluster_kit)
        if kind == "snapshot":
            time.sleep(DOWNLOAD_DELAY_SEC)
        return frame_path(self, kind, agent_id, ident)

    monkeypatch.setattr(BlobStore, "frame_path", slow_frame_path)
    config = cluster_config(tmp_path, {"checkpoint.interval": 2, "checkpoint.keep_last": 1,
                                       "checkpoint.keep_every": 0,
                                       "matchmaking.opponents": {"latest": 0.0, "snapshots": 1.0, "rivals": 0.0,
                                                                 "anchors": 0.0},
                                       "rollout.match_refresh_interval_sec": 0.5,
                                       "training.total_timesteps": 10**9})
    with cluster(tmp_path, config, [dict(name="a", cpu_learners=1, workers=1)]) as (hub, (a,)):
        assert wait_until(lambda: hub_counter(hub, "snapshot_holds") >= 3, 90), hub_counter(hub, "snapshot_holds")
        assert wait_until(lambda: len([r for r in journal_rows(hub)
                                       if any(s.network_id.startswith("ckpt_v") for s in r.seats)]) >= 50, 60)
        assert a.agent.counters()["lineup_fetch_failed"] == 0      # every granted snapshot could be downloaded
        assert hub.stop() == 0
    assert not (hub.root / "checkpoints" / EVICTED_DIR).exists()
    assert any(s.network_id.startswith("ckpt_v") for r in journal_rows(hub) for s in r.seats)
