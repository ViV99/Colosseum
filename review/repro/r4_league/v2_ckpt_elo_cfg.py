"""Checkpoint manager restart collision, ELO properties, config override semantics."""
import os
import random
import sys
import tempfile

import torch

sys.path.insert(0, "/home/viv/dev/repos/Colosseum")
from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.coordinator.ratings import EloRating
from colosseum.core.config import ColosseumConfig, load_config

print("=== A. Second run in same checkpoint dir (or resume_from a .pt => policy_version restarts at 0) ===")
with tempfile.TemporaryDirectory() as tmp:
    m1 = CheckpointManager(tmp, pool_size=3)
    for v in (100, 200, 300):
        m1.save("agent_0", v, {"w": torch.full((1,), float(v))})
    # restart: new process, new manager scans the dir
    m2 = CheckpointManager(tmp, pool_size=3)
    m2.save("agent_0", 100, {"w": torch.full((1,), -1.0)})  # new run's first ckpt
    print("index after new save:", [c.checkpoint_id for c in m2.list_checkpoints("agent_0")])
    print("on disk:", sorted(os.listdir(os.path.join(tmp, "agent_0"))))
    try:
        print("load new ckpt_v100:", m2.load("agent_0", "ckpt_v100"))
    except FileNotFoundError as e:
        print("LOAD FAILED:", e)

print("\n=== B. ELO: order dependence / N-player effective K / unbounded drift ===")
e1, e2 = EloRating(), EloRating()
games = [("a", "b")] * 10 + [("b", "a")] * 10
for w, l in games:
    e1.update(w, l)
for w, l in reversed(games):
    e2.update(w, l)
print("same 10-10 record, order A:", {k: round(v, 1) for k, v in e1.all_ratings.items()},
      " order B:", {k: round(v, 1) for k, v in e2.all_ratings.items()})
# N-player: one FFA win among 8 players -> 7 pairwise updates of K=32
e = EloRating()
for o in "bcdefgh":
    e.update("a", o)
print("one 8-player FFA win moves winner by", round(e.get("a") - 1200, 1), "points (2p win: 16)")
# Coordinator.EloRating() is constructed with no args -> k_factor is not configurable
import inspect
from colosseum.coordinator import coordinator as cmod
print("Coordinator ELO construction:", [l.strip() for l in inspect.getsource(cmod).splitlines() if "EloRating(" in l])

print("\n=== C. Per-agent override semantics ===")
cd = load_config("/home/viv/dev/repos/Colosseum/configs/examples/tic_tac_toe_multi.yaml").model_dump()
print("global lr_schedule:", cd["algorithm"]["lr_schedule"], "global batch_chunks:", cd["learner"]["batch_chunks"],
      "queue_size:", cd["learner"]["queue_size"])
cd["agents"] = {
    "beta": {"algorithm": {"learning_rate": 1e-4}, "learner": {"device": "cpu"}},
    "bot": {"type": "scripted", "scripted_class": "my.Bot"},  # what a user might try
}
cfg = ColosseumConfig(**cd)
b = cfg.get_agent_config("beta")
print("beta lr:", b.algorithm.learning_rate, "lr_schedule:", b.algorithm.lr_schedule.value,
      "batch_chunks:", b.learner.batch_chunks, "queue_size:", b.learner.queue_size)
print("trainable ids:", cfg.get_trainable_agent_ids(), "-> 'bot' silently becomes a trainable agent")
cd2 = dict(cd)
cd2["self_play"] = dict(cd["self_play"], pfsp_weighting="variance", latset_prob=0.9)
print("unknown keys accepted silently:", ColosseumConfig(**cd2).self_play)
