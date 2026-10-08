"""Eval runs a recurrent model exactly like a manual step loop with carried state (T7.1)."""
import pytest
import torch

from colosseum.eval import play_matches
from colosseum.networks.model import act
from helpers import MaskedToyEnv, MoveCounterModel, make_simple_model

STEPS = 2 * MaskedToyEnv.EPISODE_LENGTH


def _model(kind: str):
    if kind == "lstm":
        return make_simple_model(obs_dim=4, num_actions=4, core="lstm", seed=0)
    return MoveCounterModel(4)


def _manual_actions(model, carry: bool = True, reset: bool = True) -> list[int]:
    """Reference: one env, ``act()`` per step with the state carried (``carry``)
    and reset to ``initial_state(1)`` at episode end (``reset``)."""
    manual_env = MaskedToyEnv()
    obs, info = manual_env.reset(seed=7)
    state = model.initial_state(1)
    model.eval()
    for _ in range(STEPS):
        mask = torch.as_tensor(info[0]["action_mask"])[None]
        with torch.no_grad():
            out = act(model, torch.as_tensor(obs[0], dtype=torch.float32)[None], state, mask, deterministic=True)
        state = out.state if carry else model.initial_state(1)
        obs, _reward, terminated, truncated, info = manual_env.step({0: int(out.actions[0])})
        if terminated[0] or truncated[0]:
            obs, info = manual_env.reset()
            if reset:
                state = model.initial_state(1)
    return manual_env.received


def _engine_actions(model) -> list[int]:
    envs = []

    def env_fn():
        env = MaskedToyEnv()
        envs.append(env)
        return env

    records = play_matches({"rnn": model}, env_fn, [("rnn",), ("rnn",)],
                           num_envs=1, deterministic=True, seed=7)
    assert len(records) == 2
    assert all(r.length == MaskedToyEnv.EPISODE_LENGTH for r in records)
    return [a for env in envs for a in env.received]


@pytest.mark.parametrize("kind", ["lstm", "counter"])
def test_recurrent_agent_matches_manual_step_loop(kind):
    model = _model(kind)
    engine_actions = _engine_actions(model)
    assert len(engine_actions) == STEPS
    assert engine_actions == _manual_actions(model)


def test_manual_reference_detects_dropped_or_leaked_state():
    """The comparison above is sensitive: with the counter model, a loop that
    drops the state between steps, or carries it across episodes, acts differently."""
    model = _model("counter")
    engine_actions = _engine_actions(model)
    assert engine_actions != _manual_actions(model, carry=False)
    assert engine_actions != _manual_actions(model, reset=False)
