"""Learner generations on the node side (SP5 T4.1): seeds per generation (W04-21), no final snapshot for a
superseded learner (``send_final``, P-PF3-19), the cache's handover when an agent's learner leaves the machine
(``WeightCacheWriter.release_latest``)."""
from __future__ import annotations

import threading

import numpy as np

from colosseum.core.config import LearnerConfig
from colosseum.learner.learner import learner_process
from colosseum.net.chunks import WeightsMeta, encode_weights
from colosseum.node.learner_main import NO_FINAL_REASONS
from colosseum.node.weights import WeightCacheReader, WeightCacheWriter, cache_prefix
from colosseum.utils.seeding import derive_seed, learner_seed
from game_helpers import NumpyOnlyQueue, learner_appo, queue_learner_io


def test_generation_one_keeps_the_sp4_seed_and_later_ones_get_their_own():
    assert learner_seed(7, 2) == learner_seed(7, 2, 1) == derive_seed(7, "learner", 2)
    seeds = {learner_seed(7, 2, g) for g in (1, 2, 3)}
    assert len(seeds) == 3 and learner_seed(None, 2, 5) is None


def _run(send_final):
    stop, ckq = threading.Event(), NumpyOnlyQueue()
    stop.set()
    learner_process(agent_id="a", algorithm_factory=learner_appo,
                    **queue_learner_io(NumpyOnlyQueue(), weight_queues=[NumpyOnlyQueue(maxsize=1)],
                                       checkpoint_queue=ckq),
                    config=LearnerConfig(batch_chunks=2, device="cpu"), stop_event=stop, checkpoint_interval=100,
                    send_final=send_final)
    return ckq


def test_a_superseded_learner_sends_no_final_snapshot():
    assert _run(lambda: False).empty()
    assert _run(lambda: True).get_nowait()["final"] is True
    assert _run(None).get_nowait()["final"] is True
    assert set(NO_FINAL_REASONS) == {"superseded", "hub_lost"}


def _frame(version):
    return encode_weights({"w": np.full(3, float(version), np.float32)},
                          WeightsMeta("a", "latest", generation=1, policy_version=version))


def test_release_latest_hands_the_agent_back_to_the_next_writer(tmp_path):
    """A learner left unannounced versions above what the node will fetch; once its learner left the machine the
    node removes the pointer and every latest segment, so a lower fetched version is published at once."""
    prefix = cache_prefix("ab" * 16, "01" * 16, 1)
    learner, node = WeightCacheWriter(tmp_path, prefix), WeightCacheWriter(tmp_path, prefix)
    reader = WeightCacheReader(tmp_path, prefix)
    for version in (17, 18):
        learner.publish_latest("a", _frame(version), version=version, generation=1, teacher_active=False)
    held = reader.read_latest("a")
    node.release_latest("a")
    assert reader.latest_version("a") is None
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(f"{prefix}-a")]
    assert held.meta.policy_version == 18 and np.all(held.state_dict["w"] == 18.0)   # a mapped reader keeps reading
    node.publish_latest("a", _frame(16), version=16, generation=2, teacher_active=False)
    assert reader.latest_version("a") == (16, 2)
    node.release_latest("missing")                                          # nothing to remove: no error
