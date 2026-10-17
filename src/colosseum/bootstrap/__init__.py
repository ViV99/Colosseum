"""Joining machines (SP5 spec block 2): the code bundle, the node requirements, the torch build choice, the
installer script and the thin ``join`` loader. ``bundle``, ``torch_select`` and ``loader`` are standalone modules
(standard library at module level, no colosseum imports): the installer embeds them."""
