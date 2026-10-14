"""Players other than the latest weights of trainable agents (SP3 spec blocks 1-3).

``ScriptedBot`` is the base class of rule-based bots; ``RandomBot`` plays a uniformly random legal
action. ``colosseum.players.registry`` turns the config's scripted and frozen agents into picklable
specs (``BotSpec``, ``FrozenSpec``) and builds them.
"""

from colosseum.players.scripted import RandomBot, ScriptedBot

__all__ = ["RandomBot", "ScriptedBot"]
