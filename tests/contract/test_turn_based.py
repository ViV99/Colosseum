"""Turn-based transitions through the real RolloutLoop (T3.2; spec §3 contract test).

AlternatingWinEnv: player t%2 moves; the last mover wins (+1) and the other loses
(-1) on the same step. The loser's last recorded transition must carry -1 and
done=True, the winner's +1 and done=True, and non-acting steps produce no
transitions. ProbeModel(stateful) has value = number of the slot's own earlier
moves in the episode, which exposes any state advance on non-acting steps.
"""
import numpy as np
import pytest
import torch

from colosseum.core.errors import EnvContractError
from dataflow_helpers import (
    AlternatingWinEnv,
    BadMaskEnv,
    EnvFactory,
    ProbeModel,
    make_loop,
    reference_transitions,
    slot_transitions,
)


def _run(num_envs=2, steps=40, chunk_length=2):
    created = []

    def model_factory():
        model = ProbeModel(state_coef=1.0, stateful=True)
        created.append(model)
        return model

    envs = EnvFactory(AlternatingWinEnv)
    loop, col = make_loop(envs, model_factory, num_envs=num_envs, chunk_length=chunk_length)
    for _ in range(steps):
        loop.step()
    return col, envs, created


def _moves_before(transitions: list[dict], idx: int) -> int:
    """Number of earlier transitions in the same episode as transitions[idx]."""
    ep = transitions[idx]["ep"]
    return sum(1 for tr in transitions[:idx] if tr["ep"] == ep)


def test_final_rewards_and_dones_reach_both_players():
    col, _, _ = _run()
    got = slot_transitions(col.chunks)

    def episode(player, ep):
        return [(t["t"], t["reward"], t["done"]) for t in got[(0, player)] if t["ep"] == ep]

    # 3-step episode: p0 moves at t0, t2 and wins; p1 moves at t1 and loses.
    assert episode(0, 0) == [(0, 0.0, False), (2, 1.0, True)]
    assert episode(1, 0) == [(1, -1.0, True)]
    # 4-step episode: p1 gets +0.5 before its first move, then wins at t3.
    assert episode(0, 1) == [(0, 0.0, False), (2, -1.0, True)]
    assert episode(1, 1) == [(1, 0.5, False), (3, 1.0, True)]


def test_chunks_match_the_reference_and_state_advances_only_on_own_moves():
    col, envs, created = _run()
    got = slot_transitions(col.chunks)
    for env_id in range(2):
        ref = reference_transitions(envs.created[env_id].log, 2)
        for p in range(2):
            mine = got[(env_id, p)]
            expected = ref[p][: len(mine)]
            assert [(r["ep"], r["t"], r["action"], r["done"]) for r in mine] == \
                   [(r["ep"], r["t"], r["action"], r["done"]) for r in expected]
            assert np.allclose([r["reward"] for r in mine], [r["reward"] for r in expected])
            for idx, tr in enumerate(mine):
                assert tr["value"] == pytest.approx(float(_moves_before(mine, idx)))
    # inference only for acting slots: one acting slot per env per step
    assert created[0].calls == [2] * 40


def test_initial_state_and_bootstrap_follow_the_slot_own_moves():
    col, envs, _ = _run(num_envs=1, steps=60, chunk_length=2)
    ref = reference_transitions(envs.created[0].log, 2)
    offset = {0: 0, 1: 0}
    for chunk in col.chunks:
        p = int(chunk.observations[0, 3])
        k = offset[p]
        offset[p] += chunk.chunk_length
        # state before the chunk's first move = number of the slot's earlier moves
        assert torch.equal(chunk.initial_state["n"], chunk.values[:1].view(1, 1))
        if bool(chunk.dones[-1]):
            assert float(chunk.bootstrap_value) == 0.0
        else:
            assert float(chunk.bootstrap_value) == pytest.approx(float(_moves_before(ref[p], k + 2)))


def test_empty_mask_on_an_acting_slot_raises():
    loop, _ = make_loop(EnvFactory(BadMaskEnv), ProbeModel, num_envs=1)
    loop.step()
    with pytest.raises(EnvContractError, match=r"env 0, slot 1, episode step 1"):
        loop.step()


def test_non_acting_slots_send_the_zero_action():
    _, envs, _ = _run()
    acting_actions = []
    for env in envs.created:
        for step in env.log:
            for p in range(2):
                if step["active"][p]:
                    acting_actions.append(step["actions"][p])
                else:
                    assert step["actions"][p] == 0, step
    assert any(a != 0 for a in acting_actions)   # sampling would expose a non-zero leak


def test_pending_reward_of_a_never_acting_slot_is_dropped_at_episode_end():
    # Episode 0 lasts 1 step: only p0 moves and wins, so p1 gets -1 without acting.
    # Episode 1 (4 steps) gives p1 +0.5 before its first move: that transition must
    # carry +0.5 only, not the -1 left over from episode 0.
    envs = EnvFactory(AlternatingWinEnv, lengths=(1, 4))
    loop, col = make_loop(envs, ProbeModel, num_envs=1, chunk_length=2)
    for _ in range(6):        # episodes 0 (1 step), 1 (4 steps), 2 (1 step)
        loop.step()
    got = slot_transitions(col.chunks)
    assert [(t["ep"], t["t"], t["reward"], t["done"]) for t in got[(0, 1)]] == \
           [(1, 1, 0.5, False), (1, 3, 1.0, True)]
    assert [(t["ep"], t["t"], t["reward"], t["done"]) for t in got[(0, 0)]] == \
           [(0, 0, 1.0, True), (1, 0, 0.0, False), (1, 2, -1.0, True), (2, 0, 1.0, True)]
    assert loop.stats["dropped_reward_episodes"] == 2
