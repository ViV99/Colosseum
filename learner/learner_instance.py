import uuid
from time import sleep
from dataclasses import dataclass

import torch.multiprocessing as mp
from tensordict import TensorDict
from multiprocessing.connection import Connection
from requests.exceptions import HTTPError

from learner.clients import get_arena_client, ActorClient
from learner.models import IModel
from learner.replay import InferenceReplay, TrainReplay, ReplayEndReason


@dataclass
class InferenceRunner:
    process: mp.Process
    model_pipe: Connection


class Learner:
    def __init__(
        self,
        inference_proc_count: int,
        batch_size: int,
        model_type: type[IModel],
        inference_idle: float = 0.1,
        train_idle: float = 0.1
    ):
        self.inference_idle = inference_idle
        self.train_idle = train_idle
        self.id = str(uuid.uuid4())
        self.model = model_type()
        self.batch_size = batch_size
        self._storage: list[TrainReplay] = list()
        self._queue = mp.Queue()
        self._inference_runners: list[InferenceRunner] = []

        self._init_inference_runners(inference_proc_count, model_type)

    def _init_inference_runners(self, inference_proc_count: int, model_type: type[IModel]):
        mp.set_start_method("spawn")
        for _ in range(inference_proc_count):
            recv_conn, send_conn = mp.Pipe(False)
            process = mp.Process(
                target=Learner._inference_proc, args=(self.id, model_type, recv_conn, self._queue, self.inference_idle)
            )
            self._inference_runners.append(InferenceRunner(process, send_conn))

    @staticmethod
    def _update_model_state_dict(model: IModel, model_pipe: Connection):
        state_dict = None
        while model_pipe.poll():
            state_dict = model_pipe.recv()
        if state_dict:
            model.load_state_dict(state_dict)

    @staticmethod
    def _inference_proc(
        learner_id: str, model_type: type[IModel], model_pipe: Connection, output_queue: mp.Queue, idle: float
    ):
        replay = None
        actor_client = None
        model = model_type()
        while True:
            Learner._update_model_state_dict(model, model_pipe)

            if replay is None:
                replay = InferenceReplay(str(uuid.uuid4()), model.calc_rewards)
                actor_url = get_arena_client().get_actor(learner_id=learner_id, player_id=replay.id)
                actor_client = ActorClient(actor_url)
            try:
                state = actor_client.get_state(replay.id)
                state_tensor = model.state_to_tensor(state["state"])
            except HTTPError:  # not your turn
                sleep(idle)
                continue

            if state["status"] in ["ENDED_BY_ENVIRONMENT", "ENDED_BY_PLAYER_TIMEOUT"]:
                final_dict = replay.finalize(
                    state_dict=state["state"], end_reason=ReplayEndReason.ORDINARY, scores=state["scores"]
                )
                output_queue.put(final_dict)
                replay = None
                actor_client = None
            elif state["status"] == "ALIVE":
                action, logprob, state_value = model.act(replay)
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

    def _send_state_dict_to_inference(self):
        self.model.share_memory()
        for runner in self._inference_runners:
            runner.model_pipe.send(self.model.state_dict())

    def run(self):
        self._send_state_dict_to_inference()
        for runner in self._inference_runners:
            runner.process.start()

        while True:
            self._storage.append(self._queue.get(block=True))

            if len(self._storage) < self.batch_size:
                sleep(self.train_idle)
                continue

            self.model.update(self._storage)
            self._storage.clear()
            self._send_state_dict_to_inference()
