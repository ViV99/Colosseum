"""Experiments: team games (2v2), FFA ratings, matchmaking seat/team handling."""
from __future__ import annotations

import random
from collections import Counter

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from colosseum.envs.base_env import BaseEnv
from harness import make_net
from toy_envs import CatPolicy, MLPEncoder, ValueHead

random.seed(0)
torch.manual_seed(0)


def section(name):
    print("\n" + "=" * 70 + f"\n{name}\n" + "=" * 70)


class TeamEnv(BaseEnv):
    """4 seats; team 0 = seats {0,1}, team 1 = seats {2,3}. One step. Each seat
    plays 0/1; team with larger sum wins; outcome reported per seat."""

    @property
    def num_players(self):
        return 4

    @property
    def observation_space(self):
        return gymnasium.spaces.Box(0, 1, shape=(4,), dtype=np.float32)

    @property
    def action_space(self):
        return gymnasium.spaces.Discrete(2)

    def reset(self, seed=None):
        return {p: np.eye(4, dtype=np.float32)[p] for p in range(4)}, {p: {} for p in range(4)}

    def step(self, actions):
        s0 = int(actions[0]) + int(actions[1])
        s1 = int(actions[2]) + int(actions[3])
        o0 = 1.0 if s0 > s1 else (0.0 if s0 < s1 else 0.5)
        out = {0: o0, 1: o0, 2: 1 - o0, 3: 1 - o0}
        obs, _ = self.reset()
        return (obs, {p: out[p] - 0.5 for p in range(4)}, {p: True for p in range(4)},
                {p: False for p in range(4)}, {p: {"outcome": out[p]} for p in range(4)})


def biased_net(action):
    net = make_net(MLPEncoder(4), CatPolicy(32, 2), ValueHead(32))
    with torch.no_grad():
        net.policy.head.weight.zero_()
        net.policy.head.bias.copy_(torch.tensor([-20.0, 20.0] if action == 1 else [20.0, -20.0]))
    return net


section("EVAL 2v2: agent A always plays 1 (strictly better), agent B always plays 0")
from colosseum.eval import evaluate_agents
fac = lambda: make_net(MLPEncoder(4), CatPolicy(32, 2), ValueHead(32))
mat = evaluate_agents({"A": {"state_dict": biased_net(1).state_dict()},
                       "B": {"state_dict": biased_net(0).state_dict()}},
                      TeamEnv, fac, num_matches=600, num_envs=8)
print(mat.summary())
r = mat.get("A", "B")
print(f"True A-vs-B team win rate is 1.0; reported win_rate_A={r.win_rate_a:.2f}, draws={r.draws}/{r.num_matches}")

section("PFSP arena match for a 4-seat game (2 trainable agents)")
from colosseum.coordinator.agent_pool import AgentPool
from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.coordinator.matchmaker import PFSPMatchmaker
from colosseum.coordinator.ratings import WinRateTracker
import tempfile
pool = AgentPool()
pool.register_trainable("a")
pool.register_trainable("b")
mm = PFSPMatchmaker(pool, CheckpointManager(tempfile.mkdtemp(), pool_size=4), WinRateTracker(),
                    self_play_ratio=0.0)
mc = mm.generate_matches("a", 1, 4)[0]
print("arena seats:", [s.agent_id for s in mc.player_slots],
      "-> with team={0,1} vs {2,3} both teams are mixed {a,b}")
seat0 = Counter(mm.generate_matches("a", 200, 2)[i].player_slots[0].agent_id for i in range(200))
print("2-player arena, who sits in seat 0 (launcher always calls with primary agent 'a'):", dict(seat0))
mc = mm.generate_matches("a", 1, 1)[0]
print("num_players=1 'arena' match seats:", [s.agent_id for s in mc.player_slots])
mc3 = mm.generate_matches("a", 1, 3)[0]
print("3-player FFA arena seats:", [s.agent_id for s in mc3.player_slots],
      "(only 2 distinct agents ever; 'a' always 2 seats)")

section("Coordinator ratings: mixed-team arena result and FFA ranks")
from colosseum.coordinator.coordinator import Coordinator
from colosseum.core.config import ColosseumConfig, EnvConfig, NetworkConfig
from colosseum.core.types import MatchResult
from colosseum.core.outcomes import player_outcomes
cfg = ColosseumConfig(env=EnvConfig(env_class="x.Y", num_players=4),
                      networks=NetworkConfig(encoder_class="a.b", policy_class="a.b", value_class="a.b"),
                      checkpoint={"dir": tempfile.mkdtemp()})
coord = Coordinator(cfg)
# seats a,b,a,b; team {0,1} wins: seat outcomes 1,1,0,0
res = MatchResult("m", player_outcomes={"a:latest": 1.0, "b:latest": 1.0},  # dict keys collide!
                  total_rewards={})
outs = [1.0, 1.0, 0.0, 0.0]
keys = ["a:latest", "b:latest", "a:latest", "b:latest"]
po = {}
for k, o in zip(keys, outs):
    po[k] = o  # what _report_episode_result does: later seats overwrite earlier
print("worker-side player_outcomes dict for seats a,b,a,b (team0 wins):", po)
coord.report_match_result(MatchResult("m", player_outcomes=po))
print("ELO after:", coord.elo.all_ratings, " winrate a vs b:", coord.win_rates.get_win_rate("a", "b"))

coord2 = Coordinator(cfg)
for _ in range(50):
    o = player_outcomes([0, 0, 0], {0: {"rank": 1}, 1: {"rank": 2}, 2: {"rank": 3}}, 3)
    coord2.report_match_result(MatchResult("m", player_outcomes={"x:latest": o[0], "y:latest": o[1], "z:latest": o[2]}))
print("FFA ranks x=1,y=2,z=3 x50 -> ELO:", {k: round(v) for k, v in coord2.elo.all_ratings.items()})
print("FFA rank->outcome for 4 players ranks 1,2,2,4:",
      player_outcomes([0] * 4, {0: {"rank": 1}, 1: {"rank": 2}, 2: {"rank": 2}, 3: {"rank": 4}}, 4))
print("solo outcome from rewards [7.0]:", player_outcomes([7.0]))

section("Worker result reporting when one agent holds several seats (FFA a,b,a)")
from harness import run_worker
from toy_envs import FFAElimEnv


class FFARanked(FFAElimEnv):
    RANK = {0: 1, 1: 2, 2: 3}   # seat0 wins, seat1 second, seat2 last


fac3 = lambda: make_net(MLPEncoder(5), CatPolicy(32, 3), ValueHead(32))
_, results = run_worker(lambda: FFARanked("active"), fac3, steps=12, num_envs=1, chunk_length=6,
                        slot_agent_map=[["a", "b", "a"]], agent_ids=("a", "b"))
r = results[0]
print("seat ranks: a(seat0)=1, b(seat1)=2, a(seat2)=3 -> fair per-agent mean: a=0.5, b=0.5")
print("reported player_outcomes:", r.player_outcomes, " total_rewards:", r.total_rewards)
coord3 = Coordinator(cfg)
coord3.report_match_result(r)
print("ELO after one such match:", {k: round(v, 1) for k, v in coord3.elo.all_ratings.items()})
