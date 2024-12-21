import uuid
from time import sleep
from dataclasses import dataclass

import torch.multiprocessing as mp
from multiprocessing.connection import Connection
from requests.exceptions import HTTPError

from learner.clients import get_arena_client, ActorClient
from learner.models import IModel
from learner.replay import Replay, ReplayEndReason


@dataclass
class InferenceRunner:
    process: mp.Process
    model_pipe: Connection


class Learner:
    def __init__(self, inference_proc_count: int, batch_size: int, model_type: type[IModel]):
        self.id = str(uuid.uuid4())
        self.model = model_type()
        self.batch_size = batch_size
        self._storage: dict[str, Replay] = dict()
        self._queue = mp.Queue()
        self._inference_runners: list[InferenceRunner] = []

        self._init_inference_runners(inference_proc_count)

    def _init_inference_runners(self, inference_proc_count: int):
        mp.set_start_method("spawn")
        for _ in range(inference_proc_count):
            recv_conn, send_conn = mp.Pipe(False)
            process = mp.Process(target=Learner._inference_proc, args=(recv_conn, self._queue))
            self._inference_runners.append(InferenceRunner(process, send_conn))

    @staticmethod
    def _update_model_state_dict(model: IModel, model_pipe: Connection):
        state_dict = None
        while model_pipe.poll():
            state_dict = model_pipe.recv()
        if state_dict:
            model.load_state_dict(state_dict)

    @staticmethod
    def _inference_proc(learner_id: str, model_pipe: Connection, output_queue: mp.Queue):
        replay = None
        actor_client = None
        model = None
        while True:
            Learner._update_model_state_dict(model, model_pipe)
            if model is None:
                continue
            if replay is None:
                replay = Replay(str(uuid.uuid4()), model.calc_rewards)
                actor_url = get_arena_client().get_actor(learner_id=learner_id, player_id=replay.id)
                actor_client = ActorClient(actor_url)
            try:
                state = actor_client.get_state(replay.id)
                state_tensor = model.state_to_tensor(state["state"])
            except HTTPError:  # not your turn
                sleep(0.1)
                continue

            if state["status"] in ["ENDED_BY_ENVIRONMENT", "ENDED_BY_PLAYER_TIMEOUT"]:
                final_dict = replay.finalize(
                    state_dict=state["state"], end_reason=ReplayEndReason.ORDINARY, scores=state["scores"]
                )
                output_queue.put(final_dict)
                replay = None
                actor_client = None
            elif state["status"] == "ALIVE":
                action, logprob, state_value = model.act(state_tensor)
                action_dict = model.action_to_dict(action)
                try:
                    actor_client.turn(action_dict)
                except HTTPError:  # invalid move
                    final_dict = replay.finalize(
                        state_dict=state["state"], end_reason=ReplayEndReason.WRONG_MOVE
                    )
                    output_queue.put(final_dict)
                    replay = None
                    actor_client = None
                else:
                    replay.update_state(state["state"], state_tensor, action, logprob, state_value)

    def run(self):
        while True:
            finalized_replays = [replay for replay in self._storage.values() if replay.is_ended]
            if len(finalized_replays) < self.batch_size:
                sleep(0.1)

            self.model.update(finalized_replays)
            for replay in finalized_replays:
                self._storage.pop(replay.id)
