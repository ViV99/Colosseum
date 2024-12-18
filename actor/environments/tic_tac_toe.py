from actor.environments import IEnvironment
from actor.error import ActorError


class TicTacToeEnvironment(IEnvironment):
    _WIN_MASKS = [7, 56, 448, 73, 146, 292, 273, 96]

    def __init__(self, id: str, players_ids: list[str]):
        if len(players_ids) != 2:
            raise ActorError(type="BAD_PLAYERS_COUNT", code=400)

        self._players_masks = [0, 0]

        super().__init__(id, players_ids)

    @classmethod
    def get_players_count(cls) -> int:
        return 2

    def is_turn_available(self, player_id: str) -> bool:
        full_mask = self._players_masks[0] | self._players_masks[1]

        return self._players_ids[full_mask.bit_count() % 2] == player_id

    def _turn(self, player_id: str, position: int) -> None:
        full_mask = self._players_masks[0] | self._players_masks[1]
        if (full_mask & (1 << position)) != 0:
            raise ActorError(type="POSITION_ALREADY_OCCUPIED", code=400)

        self._players_masks[full_mask.bit_count() % 2] |= 1 << position

    def _get_state(self, player_id: str) -> list[str]:
        rows = []
        for i in range(3):
            rows.append("".join([self._get_symbol_by_position(i, j) for j in range(3)]))
        return rows

    def get_scores(self) -> dict[str, float]:
        results = dict()
        for i in range(2):
            result = 0
            for win_mask in self._WIN_MASKS:
                if (self._players_masks[i] & win_mask) == win_mask:
                    results[i] = 1
                    break
            results[self._players_ids[i]] = result

        return results

    def is_ended(self) -> bool:
        return sum(self.get_scores().values()) > 0

    def _get_symbol_by_position(self, i: int, j: int) -> str:
        index = i * 3 + j
        if (self._players_masks[0] & (1 << index)) != 0:
            return "X"
        elif (self._players_masks[1] & (1 << index)) != 0:
            return "O"
        else:
            return "."
