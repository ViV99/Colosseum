import multiprocessing
import uuid
from time import sleep

from requests.exceptions import HTTPError

from learner.clients import get_arena_client, ActorClient
from learner.models import IModel
from learner.replay import Replay, ReplayEndReason

class Learner:
    def __init__(self, processes_count: int, batch_size: int, model_type: type[IModel]):
        self.id = str(uuid.uuid4())
        self.model = model_type()
        self.batch_size = batch_size
        self._processes = list()
        self._storage: dict[str, Replay] = dict()

        for _ in range(processes_count):
            self._processes.append(multiprocessing.Process(target=self.subprocess_func))

        for process in self._processes:
            process.start()

    def subprocess_func(self):
        replay = None
        actor_client = None
        while True:
            if replay is None:
                replay = Replay(str(uuid.uuid4()))
                self._storage[replay.id] = replay
                actor_url = get_arena_client().get_actor(learner_id=self.id, player_id=replay.id)
                actor_client = ActorClient(actor_url)

            try:
                current_state = actor_client.get_state(replay.id)
                state_tensor = self.model.state_to_tensor(current_state["state"])
            except HTTPError:
                sleep(0.1)
                continue

            if current_state["status"] in ["ENDED_BY_ENVIRONMENT", "ENDED_BY_PLAYER_TIMEOUT"]:
                replay.finalize(
                    state_tensor=state_tensor, end_reason=ReplayEndReason.ORDINARY, scores=current_state["scores"]
                )
                replay = None
                actor_client = None
            elif current_state["status"] == "ALIVE":
                action, logprob, state_value = self.model.act(state_tensor)
                action_dict = self.model.action_to_dict(action)
                try:
                    actor_client.turn(action_dict)
                except HTTPError:
                    replay.finalize(
                        state_tensor=state_tensor, end_reason=ReplayEndReason.WRONG_MOVE
                    )
                    replay = None
                    actor_client = None
                else:
                    replay.update_state(state_tensor, action, logprob, state_value)

    def run(self):
        while True:
            finalized_replays = [replay for replay in self._storage.values() if replay.is_ended]
            if len(finalized_replays) < self.batch_size:
                sleep(0.1)

            self.model.update(finalized_replays)
            for replay in finalized_replays:
                self._storage.pop(replay.id)
