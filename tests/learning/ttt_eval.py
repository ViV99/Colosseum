"""Evaluate a trained tic-tac-toe model against a uniformly random legal-move player."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

from colosseum.core.config import load_config
from colosseum.core.run_dir import RESOLVED_CONFIG_FILE
from colosseum.eval import load_eval_model
from colosseum.networks.model import PolicyModel, act
from examples.tic_tac_toe.env import TicTacToeEnv

_CKPT_DIR_RE = re.compile(r"ckpt_v(\d+)")
_OUTCOMES = ("wins", "draws", "losses")


def load_latest_model(run_root: Path, agent_id: str = "agent_0") -> PolicyModel:
    """Read-only: build the newest ``<run>/checkpoints/<agent>/ckpt_v<N>`` with ``load_eval_model``."""
    run_root = Path(run_root)
    cfg = load_config(run_root / RESOLVED_CONFIG_FILE).get_agent_config(agent_id)
    agent_dir = run_root / "checkpoints" / agent_id
    versions = {int(m.group(1)): d for d in agent_dir.iterdir()
                if d.is_dir() and (m := _CKPT_DIR_RE.fullmatch(d.name))} if agent_dir.is_dir() else {}
    assert versions, f"no checkpoints for {agent_id} in {run_root}"
    return load_eval_model(versions[max(versions)], cfg)


@torch.no_grad()
def play_vs_random(model: PolicyModel, num_games: int = 400, seed: int = 0) -> dict:
    """The agent alternates seats (seat ``game % 2``; seat 0 moves first), so it moves first in
    half of the games. Only the current player acts: the agent greedily with the action mask,
    keeping its own model state (advanced only on its own moves, as in the worker); the
    opponent uniformly among the legal cells. Returns the win rate and W/D/L per agent seat."""
    rng = np.random.default_rng(seed)
    env = TicTacToeEnv()
    per_seat = {seat: dict.fromkeys(_OUTCOMES, 0) for seat in (0, 1)}
    for game in range(num_games):
        agent_seat = game % 2
        obs, infos = env.reset(seed=seed + game)
        state = model.initial_state(1)
        while True:
            current = infos[0]["current_player"]
            assert infos[current]["active"] and not infos[1 - current]["active"]
            mask = infos[current]["action_mask"]
            if current == agent_seat:
                out = act(model, torch.as_tensor(obs[current][None], dtype=torch.float32), state,
                          action_mask=torch.as_tensor(mask[None]), deterministic=True)
                state = out.state
                action = int(out.actions.reshape(-1)[0])
            else:
                action = int(rng.choice(np.flatnonzero(mask)))
            obs, rewards, terminated, truncated, infos = env.step({current: action, 1 - current: 0})
            if terminated[0] or truncated[0]:
                r = rewards[agent_seat]
                per_seat[agent_seat][_OUTCOMES[0 if r > 0 else (1 if r == 0 else 2)]] += 1
                break
    totals = {k: per_seat[0][k] + per_seat[1][k] for k in _OUTCOMES}
    return {"win_rate": totals["wins"] / num_games, **totals, "per_seat": per_seat, "games": num_games}
