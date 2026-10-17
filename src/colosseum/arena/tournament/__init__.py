"""``colosseum tournament`` libraries (SP4 spec block 3): ``players`` (who plays, entity names), ``schedule``
(blocks, rotations, compositions, seed domains), ``engine`` (worker processes), ``stats`` (scores, the paired
stratified bootstrap, ``confirmed``), ``modes`` (gauntlet / round-robin / halving / select, the report) and ``run``
(the tournament dir, resume, the journal). Nothing here touches the training launcher; SP5's hub can call it."""
