"""Truncated episodes bootstrap V(final_obs) into the last reward (T3.3, R1-03, R2-09)."""
import pytest

from colosseum.core.errors import EnvContractError
from dataflow_helpers import EnvFactory, GridStepEnv, ProbeModel, make_loop, slot_transitions


def test_truncation_adds_the_discounted_value_of_the_final_observation():
    created = []

    def model_factory():
        # value = t (from obs) + 10 * own moves so far: a wrong observation (the
        # next episode's first obs, t=0) or a reset state would change it.
        model = ProbeModel(obs_coeffs=(0.0, 0.0, 1.0, 0.0), state_coef=10.0, stateful=True)
        created.append(model)
        return model

    gamma = 0.9
    envs = EnvFactory(GridStepEnv, lengths=(5,), truncate_every=2, const_reward=1.0)
    loop, col = make_loop(envs, model_factory, num_envs=1, chunk_length=5, gamma=gamma)
    for _ in range(20):        # episodes 0 and 2 truncated, 1 and 3 terminated
        loop.step()
    got = slot_transitions(col.chunks)
    for p in range(2):
        last = {tr["ep"]: tr["reward"] for tr in got[(0, p)] if tr["done"]}
        final_value = 5.0 + 10.0 * 5     # terminal obs t=5, state after 5 own moves
        assert last[0] == pytest.approx(1.0 + gamma * final_value)
        assert last[2] == pytest.approx(1.0 + gamma * final_value)
        assert last[1] == pytest.approx(1.0)
        assert last[3] == pytest.approx(1.0)
    # 20 inference forwards (both seats) + exactly one forward per truncated episode
    assert created[0].calls == [2] * 22


def test_truncation_uses_the_gamma_of_the_slot_agent():
    envs = EnvFactory(GridStepEnv, lengths=(3,), truncate_every=1, const_reward=1.0)
    loop, col = make_loop(envs, lambda: ProbeModel(obs_coeffs=(0.0, 0.0, 1.0, 0.0)),
                          agent_ids=("a", "b"), num_envs=1, chunk_length=3,
                          gamma={"a": 0.5, "b": 0.9}, slot_agent_map=[["a", "b"]])
    for _ in range(3):
        loop.step()
    got = slot_transitions(col.chunks)
    assert got[(0, 0)][-1]["reward"] == pytest.approx(1.0 + 0.5 * 3.0)
    assert got[(0, 1)][-1]["reward"] == pytest.approx(1.0 + 0.9 * 3.0)
    assert got[(0, 0)][-1]["done"] and got[(0, 1)][-1]["done"]


def test_truncation_without_terminal_observation_is_a_contract_error():
    loop, _ = make_loop(EnvFactory(GridStepEnv, lengths=(3,)), ProbeModel, num_envs=1)
    loop.step()                      # both slots now hold an open transition
    with pytest.raises(EnvContractError, match="terminal_observation"):
        loop._bootstrap_truncation(0, {0: {}, 1: {}})
