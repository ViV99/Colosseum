"""The learners' LR progress between the hub's Announces (SP5 T2.5; controller ruling R28): a node forwards to its
learners the last announced global env-step count plus its own workers' env steps since that announce arrived — a
lower bound of the global count that never decreases and is never ahead of it — and the hub sends a last Announce
before ``stop_run``."""
from __future__ import annotations

import random
from types import SimpleNamespace

from colosseum.net.local import LearnerEnvSteps
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.node.learners import NodeLearners
from game_helpers import make_test_config
from hub_kit import lineup_hub, node_message, open_node


class _Host:
    """The ``LearnerHost`` part ``NodeLearners`` uses to forward env steps: ``send_child`` (recorded)."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, object]] = []

    def send_child(self, name: str, message: object) -> bool:
        self.sent.append((name, message))
        return True


def _learners(*names: str) -> tuple[NodeLearners, _Host]:
    host = _Host()
    learners = NodeLearners(host)
    for name in names:
        learners._entries[name] = SimpleNamespace(name=f"learner-{name}")
    return learners, host


def _forwarded(host: _Host, learner: str = "learner-a") -> list[int]:
    return [m.env_steps for n, m in host.sent if n == learner and isinstance(m, LearnerEnvSteps)]


def test_the_estimate_adds_the_nodes_own_steps_since_the_announce_and_never_decreases():
    learners, host = _learners("a", "b")
    learners.advance(500)                                   # nothing announced yet: nothing to forward
    assert host.sent == []
    learners.forward_env_steps(1000, own=200)               # the hub's count arrives when this node had 200
    learners.advance(250)                                   # 50 own steps later
    learners.advance(250)                                   # unchanged: not sent again
    assert _forwarded(host) == _forwarded(host, "learner-b") == [1000, 1050]
    learners.forward_env_steps(1030, own=300)               # a newer count, below the estimate: the estimate stays
    assert learners.hub_env_steps == 1050 and _forwarded(host) == [1000, 1050]
    learners.advance(330)                                   # 1030 + 30
    learners.forward_env_steps(900, own=340)                # an older (smaller) announce changes nothing
    learners.advance(340)
    assert _forwarded(host) == [1000, 1050, 1060, 1070]


def test_the_estimate_is_a_monotone_lower_bound_of_the_announced_counts_plus_own_steps():
    """Random interleavings: the forwarded value never decreases and never exceeds the largest ``announced + own steps
    since its receipt`` (each a lower bound of the global count: own steps after a receipt were not in it)."""
    rng = random.Random(7)
    for _ in range(50):
        learners, host = _learners("a")
        own, global_count, receipts = 0, 0, []
        for _ in range(60):
            steps = rng.randrange(0, 40)
            own += steps
            global_count += steps + rng.randrange(0, 80)    # other nodes' steps too
            if rng.random() < 0.2:
                announced = global_count - rng.randrange(0, 60)   # the hub's count lags a little
                learners.forward_env_steps(max(1, announced), own=own)
                receipts.append((max(1, announced), own))
            else:
                learners.advance(own)
            values = _forwarded(host)
            assert values == sorted(set(values))            # strictly growing when sent
            if values:
                assert values[-1] <= max(a + own - at for a, at in receipts) <= global_count


def test_a_cleared_run_starts_the_estimate_anew():
    learners, host = _learners("a")
    learners.forward_env_steps(5000, own=100)
    learners.clear()
    learners._entries["a"] = SimpleNamespace(name="learner-a")
    learners.forward_env_steps(10, own=120)
    assert learners.hub_env_steps == 10 and _forwarded(host)[-1] == 10


def test_the_first_count_is_a_base_even_at_zero():
    """FX3: the first count a node receives (the run's ``Welcome``: 0 for a new run) is a base like any Announce, so
    the learners' progress follows the node's own steps from the run's start; before FX3 a zero count was no base and
    nothing moved until the first Announce with steps (one heartbeat for the hub to hear of them, up to one more for
    its Announce: the whole of a short run). Own steps before the receipt (an earlier run of the session) never
    count."""
    learners, host = _learners("a", "b")
    learners.forward_env_steps(0, own=300)                  # the Welcome; 300 steps of an earlier run
    assert host.sent == []
    learners.advance(450)                                   # 150 own steps of this run
    assert _forwarded(host) == _forwarded(host, "learner-b") == [150]
    learners.forward_env_steps(0, own=460)                  # a later zero count is no newer base (160, not 0)
    learners.advance(470)
    assert _forwarded(host) == [150, 160, 170]
    learners.forward_env_steps(1000, own=500)               # the first Announce with steps: a newer base
    learners.advance(510)
    assert _forwarded(host) == [150, 160, 170, 1000, 1010]


def test_a_count_the_learner_received_before_its_spawn_value_was_set_is_kept(monkeypatch):
    """FX3: the intake's reader thread runs from its construction and may read the node's ``LearnerEnvSteps`` (sent
    at the learner's control hello) before the learner body sets the spawn-time value: that value is a floor, never
    written over the newer count."""
    import colosseum.core.threads as threads_module
    import colosseum.learner.learner as learner_module
    from node_kit import _NullConn, run_learner_main

    class EarlyCount(_NullConn):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.env_steps = 700                            # read by the reader thread at once

        @property
        def value(self):                                    # ``ConnectionIntake.value``
            return self.env_steps

    seen = []
    monkeypatch.setattr(learner_module, "learner_process",
                        lambda *, progress_counter, **kw: seen.append(progress_counter.value))
    monkeypatch.setattr(threads_module, "configure_torch_threads", lambda *a, **k: None)
    config = make_test_config("turns")
    run_learner_main(monkeypatch, config, intake_stub=EarlyCount, env_steps=300)
    run_learner_main(monkeypatch, config, intake_stub=EarlyCount, env_steps=900)
    assert seen == [700, 900]


def test_the_hub_announces_the_budgets_env_steps_before_stop_run(tmp_path):
    hub = lineup_hub(make_test_config("turns"), tmp_path)
    node = open_node(hub)
    hub._budget.report(node.node_id, node.session_id, 4321)
    hub._inbox.put(node_message(node, event=pb.Event(kind="run_stopped")))   # the node is done at once
    hub._shutdown()
    kinds = [m.WhichOneof("m") for m in node.sent]
    stop = next(i for i, m in enumerate(node.sent) if m.WhichOneof("m") == "command" and m.command.kind == "stop_run")
    last = max(i for i, k in enumerate(kinds) if k == "announce")
    assert last < stop and node.sent[last].announce.env_steps == 4321
