"""Selection and observability libraries of SP4 (spec block 8: no process queues, no launcher).

- ``entities``: rating entities (``agent@v<N>``), version buckets, lineages;
- ``comparisons``: team-pair sufficient statistics of match results (``ComparisonTable``);
- ``matchlog``: the SQLite match journal (``MatchLog``, ``MatchLogReader``, ``MatchLogWriter``);
- ``bt``: Bradley-Terry fit, Laplace and bootstrap uncertainty, role groups, anchors, transitivity test;
- ``live``: the live refit loop of the training main process and the ``bt`` section of ``ratings.json``.

Later SP4 tasks add ``retention``, ``tournament``, ``pulse`` and ``report``. Callers import submodules; this
package imports nothing eagerly.
"""
