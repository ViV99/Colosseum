"""Run report and dashboard (SP4 spec block 5): ``data`` builds one JSON-ready dict per page section from a run
dir, ``render`` turns them into one self-contained HTML page, ``server`` serves the page and ``/api/<section>``
live. No module here touches process queues or the launcher (spec block 8: SP5's hub can host them)."""
