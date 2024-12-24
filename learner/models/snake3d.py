from pathlib import Path

import torch

from rl import PPO
from learner.replay import InferenceReplay, TrainReplay, ReplayEndReason
from base_model import IModel, ActResult


class Snake3d(IModel):
    def __init__(self, algorithm: PPO):
        super().__init__()
        self.algorithm = algorithm
        self.id = None

    def act(self, replay: InferenceReplay) -> ActResult:
        """
        :param replay: Sequence of states (N * [<state_dims>,])
        :return: Tuple of
                 chosen action ([,]),
                 action logprob ([,]),
                 state value ([,])
        """
        actions, action_logprobs, state_values = self.algorithm.select_action(replay.states)
        return ActResult(actions[0].squeeze(), action_logprobs[0].squeeze(), state_values.squeeze())

    def update(self, replays: list[TrainReplay]):
        states, actions, rewards, is_terminals, state_values, logprobs, = [], [], [], [], [], []
        for r in replays:
            states.append(r.states)
            actions.append(r.actions)
            rewards.append(r.rewards)
            is_terminal = torch.zeros_like(r.state_values, dtype=torch.int)
            is_terminal[-1] = 1
            is_terminals.append(is_terminal)
            state_values.append(r.state_values)
            logprobs.append(r.logprobs)

        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        is_terminals = torch.cat(is_terminals)
        state_values = torch.cat(state_values)
        logprobs = torch.cat(logprobs)
        self.algorithm.update(states, [actions], rewards, is_terminals, state_values, [logprobs])

    @staticmethod
    def _get_local_coordinates(snake_head: torch.Tensor, coords: torch.Tensor, board_size: int):
        loc = coords - snake_head + (board_size // 2)
        if torch.any(loc < 0).item() or torch.any(loc >= board_size).item():
            return None
        return tuple(loc)

    def state_to_tensor(self, state: dict, board_size: int = 40):
        self.id = state["snakes"][0]["id"]
        snake_head = torch.tensor(state["snakes"][0]["geometry"][0])
        tensor = torch.zeros((4, board_size, board_size, board_size), dtype=torch.float32)

        for ally in state["snakes"][0]:
            if ally["reviveRemainMs"] != 0:
                continue
            for segment in ally["geometry"]:
                local_coords = Snake3d._get_local_coordinates(snake_head, torch.tensor(segment), board_size)
                if local_coords:
                    tensor[0][local_coords] = 1  # Первый канал: свои змейки

        # Учёт врагов
        for enemy in state["enemies"]:
            if enemy["status"] == "dead":
                continue
            for segment in enemy["geometry"]:
                local_coords = Snake3d._get_local_coordinates(snake_head, torch.tensor(segment), board_size)
                if local_coords:
                    tensor[1][local_coords] = 1  # Второй канал: враги

        for fence in state["fences"]:
            local_coords = Snake3d._get_local_coordinates(snake_head, torch.tensor(fence), board_size)
            if local_coords:
                tensor[2][local_coords] = 1  # Третий канал: препятствия

        for food in state["food"]:
            local_coords = Snake3d._get_local_coordinates(snake_head, torch.tensor(food), board_size)
            if local_coords:
                tensor[3][local_coords] = food["points"]  # Четвёртый канал: мандарины

        for golden_food in state["specialFood"]["golden"]:
            local_coords = Snake3d._get_local_coordinates(snake_head, torch.tensor(golden_food), board_size)
            if local_coords:
                tensor[3][local_coords] *= 10  # Special golden food

        for suspicious_food in state["specialFood"]["suspicious"]:
            local_coords = Snake3d._get_local_coordinates(snake_head, torch.tensor(suspicious_food), board_size)
            if local_coords:
                tensor[3][local_coords] -= 1  # Special suspicious food

        return tensor

    def action_to_dict(self, action: torch.Tensor) -> dict:
        actions = [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ]
        return {
            "id": self.id,
            "direction": actions[action.item()]
        }

    def calc_rewards(
        self, states: list[dict], end_reason: ReplayEndReason, final_scores: dict[str, float]
    ) -> list[float]:
        return [states[i]["points"] - states[i - 1]["points"] for i in range(1, len(states))]

    def to(self, device: str):
        self.algorithm.to(device)

    def eval(self):
        self.algorithm.eval()

    def train(self):
        self.algorithm.train()

    def save(self, checkpoint_dir: Path):
        checkpoint_dir.mkdir(exist_ok=True)
        torch.save(self.state_dict(), checkpoint_dir / "algorithm.pt")

    def state_dict(self) -> dict[str, ...]:
        return self.algorithm.state_dict()

    def load(self, checkpoint_dir: Path):
        self.load_state_dict(
            torch.load(checkpoint_dir / "algorithm.pt",  map_location=lambda storage, loc: storage)
        )

    def load_state_dict(self, state_dict: dict[str, ...]):
        self.algorithm.load_state_dict(state_dict)

    def share_memory(self):
        self.algorithm.share_memory()
