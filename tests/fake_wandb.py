"""A fake ``wandb`` module for tests: records calls, never touches the network (T6.4).

Install it with ``monkeypatch.setitem(sys.modules, "wandb", FakeWandb(...))``.
"""

from __future__ import annotations

import types


class FakeWandb(types.ModuleType):
    """Records calls; mirrors the wandb functions ``WandBLogger`` uses.

    ``fail_init`` makes ``init`` raise; ``fail_log`` makes ``log`` raise; ``fail_define`` is a
    metric name whose ``define_metric`` raises. Every call is counted in ``calls``; tests
    check ``log_kwargs`` for a forbidden ``step=`` (R5-02).
    """

    def __init__(self, fail_init: bool = False, fail_log: bool = False, fail_define: str | None = None):
        super().__init__("wandb")
        self.fail_init = fail_init
        self.fail_log = fail_log
        self.fail_define = fail_define
        self.init_kwargs: dict | None = None
        self.defined: list[tuple[str, object]] = []
        self.logged: list[dict] = []
        self.log_kwargs: list[dict] = []  # keyword arguments of every log call (must never hold step=)
        self.calls = 0
        self.finished = False

    def init(self, **kwargs):
        self.calls += 1
        if self.fail_init:
            raise RuntimeError("no network")
        self.init_kwargs = kwargs
        return types.SimpleNamespace(finish=self._finish)

    def _finish(self):
        self.finished = True

    def define_metric(self, name, step_metric=None):
        self.calls += 1
        if name == self.fail_define:
            raise RuntimeError(f"define_metric({name}) failed")
        self.defined.append((name, step_metric))

    def log(self, row, **kwargs):
        self.calls += 1
        # Recorded, not asserted here: the logger catches exceptions raised by wandb calls.
        self.log_kwargs.append(dict(kwargs))
        if self.fail_log:
            raise RuntimeError("log failed")
        self.logged.append(dict(row))
