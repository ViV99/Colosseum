"""Verify matchmaking distribution as the launcher actually drives it."""
import collections
import random
import sys
import tempfile

import torch

sys.path.insert(0, "/home/viv/dev/repos/Colosseum")
from colosseum.coordinator.coordinator import Coordinator
from colosseum.core.config import load_config, ColosseumConfig
from colosseum.core.types import MatchResult
from colosseum.launcher import _derive_worker_configs

random.seed(0)
BASE = "/home/viv/dev/repos/Colosseum/configs/examples/tic_tac_toe.yaml"


def make_cfg(tmp, phase, agents, num_players=2, **sp):
    cd = load_config(BASE).model_dump()
    cd["checkpoint"]["dir"] = tmp
    cd["training"]["phase"] = phase
    cd["env"]["num_players"] = num_players
    cd["agents"] = {a: {} for a in agents}
    cd["self_play"].update(sp)
    return ColosseumConfig(**cd)


print("=== A. League with 3 trainable agents, launcher drives with primary=agents[0] ===")
with tempfile.TemporaryDirectory() as tmp:
    cfg = make_cfg(tmp, "league", ["a0", "a1", "a2"], self_play_ratio=0.5)
    co = Coordinator(cfg)
    ids = cfg.get_trainable_agent_ids()
    for a in ids:
        co.agent_pool.register_trainable(a)
    for a in ids:
        for v in (10, 20):
            co.checkpoint_manager.save(a, v, {"w": torch.zeros(1)})
    pair = collections.Counter()
    collecting_slots = collections.Counter()
    seat0 = collections.Counter()
    kinds = collections.Counter()
    for _ in range(2000):  # what launcher.launch/_refresh_worker_matches do
        for mc in co.generate_match_configs(ids[0], 8):
            ags = tuple(sorted(set(s.agent_id for s in mc.player_slots)))
            pair[ags] += 1
            seat0[mc.player_slots[0].agent_id] += 1
            for s in mc.player_slots:
                if s.collect_trajectories:
                    collecting_slots[s.agent_id] += 1
                kinds[(s.agent_id, "ckpt" if s.checkpoint_id else "latest")] += 1
    print("agent sets per match:", dict(pair))
    print("collecting slots per agent:", dict(collecting_slots))
    print("seat-0 occupant:", dict(seat0))
    print("slot kinds:", dict(kinds))

print("\n=== B. self_play phase, 2 trainable agents ===")
with tempfile.TemporaryDirectory() as tmp:
    cfg = make_cfg(tmp, "self_play", ["alpha", "beta"])
    co = Coordinator(cfg)
    ids = cfg.get_trainable_agent_ids()
    for a in ids:
        co.agent_pool.register_trainable(a)
    c = collections.Counter()
    for _ in range(200):
        for mc in co.generate_match_configs(ids[0], 8):
            for s in mc.player_slots:
                c[s.agent_id] += 1
    print("slots per agent:", dict(c))

print("\n=== C. Seat bias in solo self-play vs historical checkpoints (2p) ===")
with tempfile.TemporaryDirectory() as tmp:
    cfg = make_cfg(tmp, "self_play", ["agent_0"], latest_prob=0.2)
    co = Coordinator(cfg)
    co.agent_pool.register_trainable("agent_0")
    for v in (10, 20, 30):
        co.checkpoint_manager.save("agent_0", v, {"w": torch.zeros(1)})
    co.setup_matchmaker("agent_0")
    seat_of_learner_vs_ckpt = collections.Counter()
    for _ in range(500):
        for mc in co.generate_match_configs("agent_0", 8):
            s0, s1 = mc.player_slots
            if s0.checkpoint_id or s1.checkpoint_id:
                seat_of_learner_vs_ckpt["seat0" if s1.checkpoint_id else "seat1"] += 1
    print("seat of the learning (latest) policy in matches vs a checkpoint:", dict(seat_of_learner_vs_ckpt))

print("\n=== D. 4-player arena: slot pattern & worker-side outcome key collision ===")
with tempfile.TemporaryDirectory() as tmp:
    cfg = make_cfg(tmp, "league", ["a0", "a1", "a2", "a3"], num_players=4, self_play_ratio=0.0)
    co = Coordinator(cfg)
    ids = cfg.get_trainable_agent_ids()
    for a in ids:
        co.agent_pool.register_trainable(a)
    patterns = collections.Counter()
    for _ in range(200):
        for mc in co.generate_match_configs(ids[0], 8):
            patterns[tuple(s.agent_id for s in mc.player_slots)] += 1
    print("distinct-agent counts per 4p arena match:",
          collections.Counter(len(set(p)) for p in patterns.elements()))
    print("example patterns:", list(patterns)[:3])
    # Simulate the worker's _report_episode_result key building
    from colosseum.core.outcomes import player_outcomes
    slot_agents = ["a0", "a1", "a0", "a1"]
    slot_nets = ["latest"] * 4
    import numpy as np
    ep_rewards = np.array([10.0, 3.0, 1.0, 2.0])  # a0 in seat0 wins FFA, a0 in seat2 is last
    outs = player_outcomes(ep_rewards, None, 4)
    po = {}
    for p in range(4):
        po[f"{slot_agents[p]}:{slot_nets[p]}"] = float(outs[p])
    print("per-slot outcomes:", outs, "-> reported player_outcomes:", po)
    co.report_match_result(MatchResult(match_id="x", player_outcomes=po))
    print("win rate a0 vs a1 after a0 WON the FFA:", co.win_rates.get_win_rate("a0", "a1"),
          "ELO:", co.elo.all_ratings)

print("\n=== E. Self-play-only run: ratings never produced ===")
with tempfile.TemporaryDirectory() as tmp:
    cfg = make_cfg(tmp, "self_play", ["agent_0"])
    co = Coordinator(cfg)
    co.agent_pool.register_trainable("agent_0")
    for _ in range(100):
        co.report_match_result(MatchResult(match_id="m", player_outcomes={
            "agent_0:latest": 1.0, "agent_0:ckpt_v10": 0.0}))
    print("ELO table:", co.elo.all_ratings, "summary:", co.get_ratings_summary())

print("\n=== F. PFSP win rates are all-time (non-stationarity) ===")
with tempfile.TemporaryDirectory() as tmp:
    cfg = make_cfg(tmp, "league", ["a0", "a1", "a2"], self_play_ratio=0.0, pfsp_exponent=1.0)
    co = Coordinator(cfg)
    for a in ("a0", "a1", "a2"):
        co.agent_pool.register_trainable(a)
    # early: a0 lost 1000 times to a1; recently a0 beats a1 200/200
    for _ in range(1000):
        co.report_match_result(MatchResult("m", {"a0:latest": 0.0, "a1:latest": 1.0}))
    for _ in range(200):
        co.report_match_result(MatchResult("m", {"a0:latest": 1.0, "a1:latest": 0.0}))
    for _ in range(200):  # a0 vs a2 balanced
        co.report_match_result(MatchResult("m", {"a0:latest": 1.0, "a2:latest": 0.0}))
        co.report_match_result(MatchResult("m", {"a0:latest": 0.0, "a2:latest": 1.0}))
    print("wr(a0,a1) =", round(co.win_rates.get_win_rate("a0", "a1"), 3),
          "though the last 200 games were all wins; wr(a0,a2)=", co.win_rates.get_win_rate("a0", "a2"))
    c = collections.Counter()
    for _ in range(300):
        for mc in co.generate_match_configs("a0", 8):
            c[mc.player_slots[1].agent_id] += 1
    print("PFSP opponent picks for a0:", dict(c))

print("\n=== G. Missing checkpoint fallback: slot plays 'latest' but collect=False ===")
with tempfile.TemporaryDirectory() as tmp:
    cfg = make_cfg(tmp, "self_play", ["agent_0"])
    co = Coordinator(cfg)
    from colosseum.core.types import MatchConfig, PlayerSlot
    mc = MatchConfig("x", {}, [PlayerSlot("agent_0", None, True), PlayerSlot("agent_0", "ckpt_v999", False)])
    print(_derive_worker_configs([mc], co, ["agent_0"])[1:])

print("\n=== H. Win-rate tracker fed with ABSOLUTE N-player outcomes (not pairwise result) ===")
with tempfile.TemporaryDirectory() as tmp:
    cfg = make_cfg(tmp, "league", ["a", "b", "c", "d"], num_players=4)
    co = Coordinator(cfg)
    for a in "abcd":
        co.agent_pool.register_trainable(a)
    # (1) env gives ranks: a=2nd, b=4th, c=1st, d=3rd  -> outcomes via outcomes.py
    from colosseum.core.outcomes import player_outcomes
    import numpy as np
    outs = player_outcomes([0, 0, 0, 0], {0: {"rank": 2}, 1: {"rank": 4}, 2: {"rank": 1}, 3: {"rank": 3}}, 4)
    print("rank outcomes a,b,c,d:", [round(o, 3) for o in outs])
    co.report_match_result(MatchResult("m", {f"{x}:latest": o for x, o in zip("abcd", outs)}))
    w = co.win_rates
    print("a beat b -> wr(a,b)=", w.get_win_rate("a", "b"), " wr(b,a)=", w.get_win_rate("b", "a"))
    print("d beat b -> wr(d,b)=", w.get_win_rate("d", "b"), " wr(b,d)=", w.get_win_rate("b", "d"))
with tempfile.TemporaryDirectory() as tmp:
    cfg = make_cfg(tmp, "league", ["a", "b", "c"], num_players=3)
    co = Coordinator(cfg)
    # (2) reward fallback FFA: c wins, a and b both lose (tie)
    for _ in range(10):
        co.report_match_result(MatchResult("m", {"a:latest": 0.0, "b:latest": 0.0, "c:latest": 1.0}))
    w = co.win_rates
    print("a,b tied (both lost to c) x10 -> wr(a,b)=", w.get_win_rate("a", "b"), " wr(b,a)=", w.get_win_rate("b", "a"),
          " ELO a,b:", round(co.elo.get("a"), 1), round(co.elo.get("b"), 1))
