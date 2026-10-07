"""Generate TicTacToe expert BC data (heuristic: win > block > center > corner > side).

Expert plays both seats vs a random/expert mix; records (obs of mover, expert action).
usage: gen_ttt_expert.py OUT.pt [--games 3000]
"""
import random
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/viv/dev/repos/Colosseum")
from examples.tic_tac_toe.env import TicTacToeEnv

LINES = TicTacToeEnv.WINNING_LINES


def expert(board, me):
    opp = 3 - me
    empty = [i for i in range(9) if board[i] == 0]
    for mark in (me, opp):
        for ln in LINES:
            vals = [board[i] for i in ln]
            if vals.count(mark) == 2 and vals.count(0) == 1:
                return ln[vals.index(0)]
    if 4 in empty:
        return 4
    corners = [i for i in (0, 2, 6, 8) if i in empty]
    if corners:
        return random.choice(corners)
    return random.choice(empty)


def main():
    out = sys.argv[1]
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    random.seed(0)
    env = TicTacToeEnv()
    O, A = [], []
    for g in range(games):
        obs, _ = env.reset()
        done = False
        eps = random.random() * 0.5  # opponent noise so states are diverse
        while not done:
            cur = env._current_player
            a = expert(env._board, cur + 1)
            O.append(obs[cur].copy()); A.append(a)
            if random.random() < eps:
                a = random.choice([i for i in range(9) if env._board[i] == 0])
            obs, r, t, _, _ = env.step({cur: a, 1 - cur: 0})
            done = t[0]
    torch.save({"observations": torch.tensor(np.stack(O)), "actions": torch.tensor(A, dtype=torch.int64)}, out)
    print(f"saved {len(A)} samples to {out}")


if __name__ == "__main__":
    main()
