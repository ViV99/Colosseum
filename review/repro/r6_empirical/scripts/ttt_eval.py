"""TicTacToe evaluation: trained checkpoint vs uniform-random legal player / vs another checkpoint.

usage: ttt_eval.py CONFIG CKPT [--vs CKPT2] [--n 500] [--greedy]
Reports win/draw/loss with seat breakdown and the agent's illegal-move loss count.
"""
import argparse
import random
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/viv/dev/repos/Colosseum")
from colosseum.core.config import load_config
from colosseum.core.registry import build_network
from examples.tic_tac_toe.env import TicTacToeEnv


def load_net(cfg, path):
    net = build_network(cfg)
    if path not in (None, "init"):
        net.load_state_dict(torch.load(path, weights_only=True))
    net.eval()
    return net


MASK = False


def net_move(net, obs, greedy, board=None):
    m = torch.tensor(board == 0)[None] if (MASK and board is not None) else None
    with torch.no_grad():
        a, _, _, _ = net.act(torch.tensor(obs[None], dtype=torch.float32), action_mask=m, deterministic=greedy)
    return int(a.item())


def random_move(env):
    return random.choice([i for i in range(9) if env._board[i] == 0])


def play(env, players, greedy):
    """players: dict seat -> net or 'random'. Returns (rewards, illegal_by_seat)."""
    obs, _ = env.reset()
    illegal = None
    while True:
        cur = env._current_player
        p = players[cur]
        a = random_move(env) if p == "random" else net_move(p, obs[cur], greedy, env._board)
        if env._board[a] != 0:
            illegal = cur
        acts = {cur: a, 1 - cur: 0}
        obs, rew, term, trunc, _ = env.step(acts)
        if term[0]:
            return rew, illegal


def run(cfg, a_net, b, n, greedy, label_b):
    env = TicTacToeEnv()
    stats = {}
    for seat in (0, 1):
        w = d = l = ill = opp_ill = 0
        for _ in range(n // 2):
            players = {seat: a_net, 1 - seat: b}
            rew, illegal = play(env, players, greedy)
            r = rew[seat]
            if r > 0: w += 1
            elif r < 0: l += 1
            else: d += 1
            if illegal == seat: ill += 1
            elif illegal is not None: opp_ill += 1
        stats[seat] = (w, d, l, ill, opp_ill)
    tot = [sum(stats[s][i] for s in (0, 1)) for i in range(5)]
    m = n // 2 * 2
    print(f"vs {label_b} (greedy={greedy}, mask={MASK}, N={m}):")
    for s in (0, 1):
        w, d, l, ill, oill = stats[s]
        k = n // 2
        print(f"  seat {s} ({'X/first' if s == 0 else 'O/second'}): W {w/k:.3f} D {d/k:.3f} L {l/k:.3f}  "
              f"(agent illegal-move losses {ill}, opp illegal {oill})")
    print(f"  total: W {tot[0]/m:.3f} D {tot[1]/m:.3f} L {tot[2]/m:.3f}  agent illegal {tot[3]}/{m}")
    return tot


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("ckpt")
    ap.add_argument("--vs", default="random")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    cfg = load_config(args.config)
    MASK = args.mask
    a = load_net(cfg, args.ckpt)
    b = "random" if args.vs == "random" else load_net(cfg, args.vs)
    run(cfg, a, b, args.n, args.greedy, args.vs)
