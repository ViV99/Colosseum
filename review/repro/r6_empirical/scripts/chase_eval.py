"""ChaseEnv eval: checkpoint (seat s) vs uniform-random actions; also mean final distance to target."""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/viv/dev/repos/Colosseum")
from colosseum.core.config import load_config
from colosseum.core.registry import build_network
from colosseum.core.action_spec import ActionSpec
from examples.composite_action.env import ChaseEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()
    np.random.seed(0); torch.manual_seed(0)
    cfg = load_config(args.config)
    net = build_network(cfg)
    if args.ckpt != "init":
        net.load_state_dict(torch.load(args.ckpt, weights_only=True))
    net.eval()
    env = ChaseEnv()
    spec = ActionSpec.from_space(env.action_space)
    res = {0: [0, 0, 0], 1: [0, 0, 0]}
    dists = []
    for seat in (0, 1):
        for _ in range(args.n // 2):
            obs, _ = env.reset()
            done = False
            while not done:
                acts = {}
                with torch.no_grad():
                    a, _, _, _ = net.act(torch.tensor(obs[seat][None]), deterministic=args.greedy)
                acts[seat] = spec.decode(a.numpy()[0]) if spec.is_composite else a.numpy()[0]
                acts[1 - seat] = {"direction": np.random.randint(4), "speed": float(np.random.rand())}
                obs, rew, term, _, _ = env.step(acts)
                done = term[0]
            dists.append(float(np.linalg.norm(env._positions[seat] - env._targets[seat])))
            r = rew[seat]
            res[seat][0 if r > 0 else (2 if r < 0 else 1)] += 1
    k = args.n // 2
    for s in (0, 1):
        print(f"  seat {s}: W {res[s][0]/k:.3f} D {res[s][1]/k:.3f} L {res[s][2]/k:.3f}")
    print(f"  total W {(res[0][0]+res[1][0])/(2*k):.3f}; mean final dist to own target {np.mean(dists):.2f}")


if __name__ == "__main__":
    main()
