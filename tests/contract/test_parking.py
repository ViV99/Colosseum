"""Parking partial buffers on match re-assignment loses no transitions (T2.6, R2-12).

The real RolloutLoop runs in-process; WorkerCommands flip slot agents and
collect flags every few steps. Every recorded transition must be either in a
sent chunk or still in a buffer (a slot's or a parked one), exactly once, and
chunks must stay sequences of one agent's transitions with episode boundaries.
"""
from colosseum.core.types import WeightPayload, WorkerCommand, state_dict_to_numpy
from dataflow_helpers import EnvFactory, GridStepEnv, ProbeModel, TinyModel, make_loop

M1 = WorkerCommand(slot_agent_map=[["a", "b"], ["b", "a"]],
                   slot_network_map=[["latest", "latest"], ["latest", "latest"]],
                   collect_mask=[[True, True], [True, False]])
M2 = WorkerCommand(slot_agent_map=[["b", "b"], ["a", "a"]],
                   slot_network_map=[["latest", "latest"], ["latest", "latest"]],
                   collect_mask=[[True, False], [True, True]])


def test_reassignment_parks_buffers_without_losing_transitions():
    agents, num_envs, num_players = ("a", "b"), 2, 2
    loop, col = make_loop(EnvFactory(GridStepEnv, lengths=(3, 5)), ProbeModel,
                          agent_ids=agents, num_envs=num_envs, chunk_length=4,
                          slot_agent_map=[["a", "a"], ["b", "b"]])
    max_parked = 0
    for i in range(120):
        if i % 4 == 0:
            col.commands.append(M1 if (i // 4) % 2 == 0 else M2)
        loop.step()
        parked = loop.stats["parked_buffers"]
        # Parked buffers are resumed: at most one per (agent, slot), never unbounded growth.
        assert parked <= len(agents) * num_envs * num_players, f"step {i}: {parked} parked buffers"
        max_parked = max(max_parked, parked)
    assert max_parked > 0, "the scenario must exercise parking"
    stats = loop.stats
    for aid in agents:
        sent = sum(c.chunk_length for c in col.chunks if c.agent_id == aid)
        assert sent + stats[f"buffered_transitions/{aid}"] == stats[f"recorded_transitions/{aid}"]
    seen = set()
    resumed = 0
    for chunk in col.chunks:
        obs = chunk.observations.numpy()
        if len({(int(o[0]), int(o[3])) for o in obs}) > 1:
            resumed += 1                                  # rows of two (env, seat) slots
        for i in range(chunk.chunk_length):
            key = tuple(obs[i].astype(int).tolist())       # (env_id, ep, t, player)
            assert key not in seen, f"transition {key} sent twice"
            seen.add(key)
            if i + 1 < chunk.chunk_length:
                if bool(chunk.dones[i]):
                    # after a done the next row starts an episode (possibly another slot's)
                    assert obs[i + 1][2] == 0
                else:
                    # inside an episode the next row is the same slot's next step
                    assert (obs[i + 1][0], obs[i + 1][1], obs[i + 1][3]) == (obs[i][0], obs[i][1], obs[i][3])
                    assert obs[i + 1][2] == obs[i][2] + 1
    assert resumed > 0, "no chunk continues a parked buffer in another slot"


def test_behavior_version_is_the_version_at_the_first_transition():
    loop, col = make_loop(EnvFactory(GridStepEnv, lengths=(50,)), TinyModel, num_envs=1, chunk_length=4)
    loop.step()                                  # transition t=0 recorded with version 0
    col.weights["a"] = [WeightPayload.from_model("a", 7, TinyModel())]
    loop.step()                                  # the sync at the end of this step loads v7
    for _ in range(10):
        loop.step()
    versions = [c.behavior_policy_version for c in col.chunks if int(c.observations[0, 3]) == 0]
    assert versions[0] == 0
    assert versions[1:] and all(v == 7 for v in versions[1:])


def test_command_checkpoint_is_loaded_and_used_from_the_next_episode():
    created = []

    def model_factory():
        model = ProbeModel()
        created.append(model)
        return model

    loop, col = make_loop(EnvFactory(GridStepEnv, lengths=(2,)), model_factory, num_envs=1, chunk_length=2)
    col.commands.append(WorkerCommand(
        slot_agent_map=[["a", "a"]], slot_network_map=[["latest", "ckpt_v9"]],
        collect_mask=[[True, False]],
        new_checkpoints={"a": {"ckpt_v9": state_dict_to_numpy(ProbeModel().state_dict())}},
    ))
    loop.step()
    assert len(created) == 2 and created[1].calls == []   # loaded, not used mid-episode
    loop.step()                                          # episode 0 ends -> assignment applied
    for _ in range(4):
        loop.step()
    assert created[1].calls == [1] * 4                   # the checkpoint plays seat 1
    late = [c for c in col.chunks if int(c.observations[0, 1]) >= 1]
    assert late and all(int(c.observations[0, 3]) == 0 for c in late)   # seat 1 no longer collects


def test_parked_buffer_keeps_its_first_transition_version_across_a_weight_bump():
    """A buffer parked under v0 and resumed after the agent moved to v5 still
    reports v0: the version at the chunk's first transition."""
    stop_b = WorkerCommand(slot_agent_map=[["a", "b"]], slot_network_map=[["latest", "latest"]],
                           collect_mask=[[True, False]])
    resume_b = WorkerCommand(slot_agent_map=[["a", "b"]], slot_network_map=[["latest", "latest"]],
                             collect_mask=[[True, True]])
    loop, col = make_loop(EnvFactory(GridStepEnv, lengths=(3,)), TinyModel, agent_ids=("a", "b"),
                          num_envs=1, chunk_length=8, slot_agent_map=[["a", "b"]])
    col.commands.append(stop_b)
    for _ in range(3):
        loop.step()                       # episode 0 ends -> b's 3 transitions are parked
    assert loop.stats["parked_buffers"] == 1 and loop.stats["buffered_transitions/b"] == 3
    col.weights["b"] = [WeightPayload.from_model("b", 5, TinyModel())]
    col.commands.append(resume_b)
    loop.step()                           # loads b v5; resume_b staged
    assert loop._policy_versions["b"] == 5
    for _ in range(2):
        loop.step()                       # episode 1 ends -> seat 1 resumes the parked buffer
    assert loop.stats["parked_buffers"] == 0
    for _ in range(16):
        loop.step()
    b_chunks = [c for c in col.chunks if c.agent_id == "b"]
    assert len(b_chunks) == 2
    resumed, fresh = b_chunks
    eps = [int(o[1]) for o in resumed.observations]
    assert eps == [0, 0, 0, 2, 2, 2, 3, 3]          # parked episode 0, then episodes 2-3 under v5
    assert resumed.behavior_policy_version == 0
    assert fresh.behavior_policy_version == 5
