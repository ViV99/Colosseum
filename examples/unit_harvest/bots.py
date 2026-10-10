"""A scripted unit_harvest player (SP3 spec block 2): the port of ``game.scripted_action`` (carriers
walk home, the other workers walk to the nearest resource) whose base builds whenever the action
mask allows it (``scripted_action`` alone would also try when every unit slot is used)."""
from __future__ import annotations

from typing import Any

import numpy as np

from colosseum.players import ScriptedBot
from examples.unit_harvest.game import scripted_action


class HarvestBot(ScriptedBot):
    def act(self, obs: Any, mask: Any, info: Any) -> dict:
        action = scripted_action(obs)
        can_build = True if mask is None else bool(np.asarray(mask["base"])[1])
        return {"base": 1 if can_build else 0, "workers": np.asarray(action["workers"], dtype=np.int64)}
