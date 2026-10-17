"""Pulse (SP4 T6.1, spec block 6): the evaluation section and its checks, the feeder's choice and backlog, and the
pulse process on a queue pair (one spawn child)."""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue

import numpy as np
import pytest
import yaml

from colosseum.arena.pulse import PulseFeeder, PulseRequest, PulseResults, pulse_plan, pulse_target
from colosseum.core.config import load_config
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_model, env_spec, validate_config
from colosseum.players.registry import load_fixed_players, resolve_player_roles
from game_helpers import agent_role_of, make_test_config, scripted_agent

AGENTS = {"main": {}, "rnd": scripted_agent(), "ones": scripted_agent("game_helpers.ConstantBot", value=1)}


def _cfg(**evaluation):
    return make_test_config("turns", agents=AGENTS, evaluation={"enabled": True, **evaluation})


def _write(tmp_path, **sections) -> str:
    data = {"env": {"env_class": "game_helpers.TurnTakingGame"},
            "networks": {"model_class": "game_helpers.GameTestModel"}, "agents": AGENTS, **sections}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return str(path)


def test_the_evaluation_section_defaults_and_config_checks(tmp_path):
    ev = load_config(_write(tmp_path)).evaluation
    assert (ev.enabled, ev.agents, ev.field, ev.layouts, ev.every, ev.blocks, ev.num_envs, ev.deterministic,
            ev.max_backlog) == (False, None, None, None, 1, 25, 8, False, 4)
    with pytest.raises(ConfigError, match="match_log"):
        load_config(_write(tmp_path, evaluation={"enabled": True}, match_log={"enabled": False}))
    with pytest.raises(ConfigError, match="field"):
        load_config(_write(tmp_path, evaluation={"enabled": True, "field": []}))
    for bad in ({"every": 0}, {"blocks": 0}, {"num_envs": 0}, {"max_backlog": 0}):
        with pytest.raises(ConfigError):
            load_config(_write(tmp_path, evaluation=bad))


def test_validate_checks_the_field_agents_layouts_and_prints_the_plan():
    lines = validate_config(_cfg(blocks=5)).lines
    (line,) = [x for x in lines if x.startswith("pulse:")]
    assert "agents ['main']" in line and "field ['rnd', 'ones']" in line and "20 matches per snapshot" in line
    for evaluation, message in (({"field": ["nobody"]}, "nobody"), ({"field": ["main"]}, "scripted or frozen"),
                                ({"agents": ["rnd"]}, "trainable"), ({"layouts": ["9p"]}, "9p")):
        with pytest.raises(ConfigError, match=message):
            validate_config(_cfg(**evaluation))
    assert not [x for x in validate_config(make_test_config("turns", agents=AGENTS)).lines if x.startswith("pulse:")]


def test_role_coverage_is_checked_per_agent_and_layout():
    cfg = make_test_config("asymmetric", agents={"hunter": {"roles": ["hunter"]}, "prey": {"roles": ["prey"]},
                                                 "preybot": {**scripted_agent(), "roles": ["prey"]}},
                           evaluation={"enabled": True, "agents": ["prey"]})
    with pytest.raises(ConfigError, match="prey"):
        pulse_plan(cfg, env_spec(cfg), resolve_player_roles(cfg, env_spec(cfg)))


def _request(version, agent="main", final=False, env_steps=None):
    return PulseRequest(agent, f"ckpt_v{version}", version, env_steps if env_steps is not None else 10 * version,
                        {"w": np.zeros(2, np.float32)}, final=final)


def test_the_feeder_offers_every_kth_snapshot_and_the_final_one_and_drops_the_oldest(caplog):
    feeder = PulseFeeder(["main"], every=2, max_backlog=2)
    for version in range(1, 8):
        feeder.offer("main", f"ckpt_v{version}", version, 10 * version, {"w": np.zeros(1)}, final=False)
    feeder.offer("other", "ckpt_v1", 1, 10, {}, final=False)                 # not evaluated
    with caplog.at_level(logging.INFO, logger="colosseum.arena.pulse"):
        feeder.offer("main", "ckpt_v8", 8, 80, {}, final=True)                # 8th: even, and final
        feeder.offer("main", "ckpt_v9", 9, 90, {}, final=True)                # odd, but final
    q = queue.Queue(maxsize=1)
    assert feeder.pump(q) == 1 and q.get_nowait().version == 8           # backlog [8, 9]: the oldest goes first
    state = feeder.state()
    assert state["offered"]["main"] == [2, 4, 6, 8, 9]
    assert state["skipped"]["main"] == [2, 4, 6] and feeder.skipped == 3
    assert any("ckpt_v6 of main skipped" in r.getMessage() for r in caplog.records)
    assert "other" not in state["offered"]


def test_a_final_snapshot_dropped_by_the_backlog_is_logged_and_unevaluated(caplog):
    feeder = PulseFeeder(["main", "b"], every=1, max_backlog=1)
    feeder.offer("main", "ckpt_v5", 5, 50, {}, final=True)
    with caplog.at_level(logging.INFO, logger="colosseum.arena.pulse"):
        feeder.offer("b", "ckpt_v7", 7, 70, {}, final=True)                 # pushes main's final out
    assert any("the final snapshot ckpt_v5 of main skipped" in r.getMessage() for r in caplog.records)
    assert feeder.state()["skipped"] == {"main": [5], "b": []}
    assert sorted(feeder.unevaluated_final()) == [("b", "ckpt_v7"), ("main", "ckpt_v5")]


def test_the_feeder_keeps_no_weights_of_a_request_in_flight():
    feeder = PulseFeeder(["main"], every=1, max_backlog=2)
    feeder.offer("main", "ckpt_v1", 1, 10, {"w": np.zeros(3)}, final=False)
    q = queue.Queue(maxsize=1)
    feeder.pump(q)
    assert feeder.pending == 1 and not any(hasattr(t, "state_dict") for t in feeder._in_flight.values())


def test_the_feeder_bookkeeping_adds_up():
    feeder = PulseFeeder(["main"], every=1, max_backlog=1)
    q = queue.Queue(maxsize=1)
    for version in (1, 2, 3):
        feeder.offer("main", f"ckpt_v{version}", version, version, {}, final=version == 3)
        feeder.pump(q)
    taken = q.get_nowait()
    feeder.done(PulseResults(taken.agent_id, taken.checkpoint_id, taken.version, [], True))
    feeder.pump(q)
    state = feeder.state()
    offered, evaluated = set(state["offered"]["main"]), set(state["evaluated"]["main"])
    rest = set(state["skipped"]["main"]) | set(state["unevaluated"]["main"])
    assert evaluated == {1} and offered == evaluated | rest and not evaluated & rest
    assert state["unevaluated"]["main"] == [3] and feeder.unevaluated_final() == [("main", "ckpt_v3")]


def test_system_values_reach_every_later_system_record(tmp_path):
    import json

    from colosseum.metrics.hub import MetricsHub
    from colosseum.metrics.jsonl import MetricsWriter

    hub = MetricsHub(writer=MetricsWriter(tmp_path / "m.jsonl"), ratings_path=tmp_path / "r.json", agent_ids=["a"],
                     total_timesteps=10, log_interval=1, console_interval_sec=0.0)
    hub.set_system_value("pulse_skipped_snapshots", 2)
    hub.maybe_tick(env_steps=1, ratings={}, queue_depths={}, force=True)
    hub.set_system_value("pulse_skipped_snapshots", 3)
    hub.close(env_steps=2, ratings={}, queue_depths={})
    records = [json.loads(line) for line in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert [r["pulse_skipped_snapshots"] for r in records if r["kind"] == "system"] == [2, 3]


def test_the_pulse_process_plays_every_snapshot_on_the_same_seeds(tmp_path):
    cfg = _cfg(blocks=3, num_envs=2)
    spec = env_spec(cfg)
    plan = pulse_plan(cfg, spec, resolve_player_roles(cfg, spec))
    fixed = load_fixed_players(cfg, spec, plan.field)
    _roles, role = agent_role_of(cfg, "main")
    ctx = mp.get_context("spawn")
    requests, results, stop = ctx.Queue(maxsize=1), ctx.Queue(), ctx.Event()
    proc = ctx.Process(target=pulse_target, daemon=True, kwargs=dict(
        log_dir=str(tmp_path / "logs"), config=cfg, requests=requests, results=results, stop_event=stop,
        field_specs=fixed, field=plan.field, agents=plan.agents, run_id="0123456789abcdef"))
    proc.start()
    try:
        got = []
        for version in (5, 10):
            state = {k: v.detach().numpy().copy() for k, v in
                     build_model(cfg.get_agent_config("main"), role).state_dict().items()}
            requests.put(PulseRequest("main", f"ckpt_v{version}", version, 100 * version, state))
            got.append(results.get(timeout=60))
        requests.put(None)
        proc.join(30)
        assert proc.exitcode == 0
    finally:
        stop.set()
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        for q in (requests, results):
            q.cancel_join_thread()
            q.close()
    assert [m.version for m in got] == [5, 10] and all(m.complete for m in got)
    seeds = [sorted({r.seed for r in m.results}) for m in got]
    assert seeds[0] == seeds[1] and len(seeds[0]) == 3                     # 3 blocks: the same seeds for both
    for m in got:
        assert {r.source for r in m.results} == {"pulse"} and {r.env_steps for r in m.results} == {100 * m.version}
        assert all(r.uid.startswith(f"pulse-0123456789ab/P/2p/main@v{m.version}/") for r in m.results)
        assert len(m.results) == 2 * 3 * 2                                 # 2 field members x 3 blocks x 2 sides
    assert (tmp_path / "logs" / "pulse.log").is_file()


class _Recorder:
    """Records the calls a stand-in receives (``calls``: name, args)."""

    def __init__(self, *allowed: str) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self._allowed = set(allowed)

    def __getattr__(self, name):
        if name.startswith("_") or name not in self._allowed:
            raise AssertionError(f"unexpected call {name}")
        return lambda *args, **kwargs: self.calls.append((name, args))


def test_the_launcher_offers_saved_snapshots_and_journals_pulse_results_only(tmp_path, monkeypatch):
    """Contract amendment A20: pulse results reach the journal writer only, never the coordinator (online ELO,
    PFSP, matchmaker) or the hub's episode stats; the counters reach the system record; pulse_state.json."""
    import json

    from colosseum.launcher import Launcher
    from game_helpers import make_coordinator, make_match_result, make_test_run_dir

    cfg = _cfg()
    run = make_test_run_dir(cfg, tmp_path)
    launcher = Launcher(cfg, run)
    launcher._coordinator = make_coordinator(cfg, run.checkpoints)
    launcher._env_step_counter.add(70)
    launcher._pulse_feeder = PulseFeeder(["main"], every=1, max_backlog=4)
    launcher._pulse_requests = requests = queue.Queue(maxsize=1)
    launcher._pulse_results = "pulse results queue"
    _roles, role = agent_role_of(cfg, "main")
    model = build_model(cfg.get_agent_config("main"), role)
    state = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
    launcher._save_checkpoint({"agent_id": "main", "policy_version": 7, "final": False, "model_state": state,
                               "trainer_state_bytes": None})
    assert (run.checkpoints / "main" / "ckpt_v7" / "model.pt").is_file()

    result = make_match_result([(0, "main", "ckpt_v7", "player", 7, 7), (1, "rnd", "fixed")], source="pulse",
                               uid="pulse-x/P/2p/main@v7/rnd/0/r0", env_steps=70)
    sent: list = []
    monkeypatch.setattr(launcher, "_drain", lambda q, label, producers: list(sent) if label == "pulse-results" else [])
    launcher._match_log_writer = writer = _Recorder("submit")
    launcher._hub = hub = _Recorder("set_system_value")
    launcher._coordinator = _Recorder()                           # any coordinator call fails the test
    launcher._pump_pulse()
    taken = requests.get_nowait()
    assert (taken.agent_id, taken.checkpoint_id, taken.version, taken.env_steps) == ("main", "ckpt_v7", 7, 70)
    assert not any(hasattr(v, "detach") for v in taken.state_dict.values())  # numpy only crosses processes
    sent.append(PulseResults("main", "ckpt_v7", 7, [result], True))
    launcher._pump_pulse()
    assert writer.calls == [("submit", ([result],))]
    values = {args[0]: args[1] for name, args in hub.calls}
    assert values == {"pulse_skipped_snapshots": 0, "pulse_evaluated_snapshots": 1, "pulse_pending_snapshots": 0}
    state_file = json.loads(run.pulse_state_path.read_text())
    assert state_file["evaluated"] == {"main": [7]} and state_file["offered"] == {"main": [7]}


def _numpy_state(cfg, agent_id):
    _roles, role = agent_role_of(cfg, agent_id)
    model = build_model(cfg.get_agent_config(agent_id), role)
    return {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}


def _run_pulse_in_process(cfg, requests, monkeypatch, *, stop=None):
    """``pulse_main`` in this process on plain queues (cheap: no spawn); returns the messages it sent."""
    import threading

    from colosseum.arena import pulse as pulse_module
    from colosseum.core import threads

    monkeypatch.setattr(threads, "configure_torch_threads", lambda *a, **k: None)
    spec = env_spec(cfg)
    plan = pulse_plan(cfg, spec, resolve_player_roles(cfg, spec))
    q_in, q_out = queue.Queue(), queue.Queue()
    for request in [*requests, None]:
        q_in.put(request)
    pulse_module.pulse_main(cfg, requests=q_in, results=q_out, stop_event=stop or threading.Event(),
                            field_specs=load_fixed_players(cfg, spec, plan.field), field=plan.field,
                            agents=plan.agents, run_id="0123456789abcdef")
    out = []
    while not q_out.empty():
        out.append(q_out.get_nowait())
    return out


def test_the_pulse_plays_score_layouts_for_two_agents(monkeypatch):
    """A one-team layout (score_blocks: the homogeneous team and cross-play with the field) and two evaluated
    agents, each on its own seats and uids."""
    cfg = make_test_config("coop", agents={"a": {}, "b": {}, "rnd": scripted_agent()},
                           evaluation={"enabled": True, "blocks": 2, "num_envs": 2})
    got = _run_pulse_in_process(cfg, [PulseRequest("a", "ckpt_v3", 3, 30, _numpy_state(cfg, "a")),
                                      PulseRequest("b", "ckpt_v4", 4, 40, _numpy_state(cfg, "b"))], monkeypatch)
    assert [(m.agent_id, m.version) for m in got] == [("a", 3), ("b", 4)]
    for m in got:
        prefix = f"pulse-0123456789ab/P/coop2/{m.agent_id}@v{m.version}/"
        uids = {r.uid for r in m.results}
        # 2 blocks x (the homogeneous team, rotation 0 + cross-play with rnd on either seat, rotations 0 and 1)
        assert uids == ({f"{prefix}-/{k}/r0" for k in (0, 1)}
                        | {f"{prefix}rnd/{k}/r{seat}" for k in (0, 1) for seat in (0, 1)})
        assert {r.source for r in m.results} == {"pulse"}
        assert {s.agent_id for r in m.results for s in r.seats} == {m.agent_id, "rnd"}
    assert {r.seed for r in got[0].results} == {r.seed for r in got[1].results}


def test_unscheduled_compositions_are_logged_once_per_agent(monkeypatch, caplog):
    cfg = make_test_config("asymmetric", agents={"hunter": {"roles": ["hunter"]}, "prey": {"roles": ["prey"]},
                                                 "preybot": {**scripted_agent(), "roles": ["prey"]},
                                                 "hunterbot": {**scripted_agent(), "roles": ["hunter"]}},
                           evaluation={"enabled": True, "agents": ["hunter"], "blocks": 1, "num_envs": 2})
    state = _numpy_state(cfg, "hunter")
    with caplog.at_level(logging.INFO):
        got = _run_pulse_in_process(cfg, [PulseRequest("hunter", f"ckpt_v{v}", v, v, state) for v in (1, 2)],
                                    monkeypatch)
    assert [m.version for m in got] == [1, 2]
    messages = [(r.name, r.getMessage()) for r in caplog.records]
    # the schedule's own line comes from pulse_plan (the main process) only, never per snapshot
    assert not [m for n, m in messages if n.endswith("schedule") and ("hunter@v1" in m or "hunter@v2" in m)]
    drops = [m for n, m in messages if n == "colosseum.arena.pulse" and "hunterbot is not played" in m]
    assert len(drops) == 1


def test_put_until_stop_still_sends_after_the_stop_when_there_is_room():
    import threading

    from colosseum.arena.tournament.engine import put_until_stop

    stop = threading.Event()
    stop.set()
    q = queue.Queue(maxsize=1)
    assert put_until_stop(q, "finished", stop) and q.get_nowait() == "finished"
    q.put("occupied")
    assert not put_until_stop(q, "late", stop)


def test_play_lineups_ends_early_when_asked_to_stop():
    from colosseum.core.registry import make_env
    from colosseum.core.types import Lineup, SeatAssignment
    from colosseum.eval import play_lineups
    from colosseum.players import RandomBot

    cfg = make_test_config("turns", agents=AGENTS)
    lineups = [Lineup("2p", [SeatAssignment("x"), SeatAssignment("y")], uid=f"u{i}") for i in range(4)]
    kwargs = dict(env_fn=lambda: make_env(cfg), models={"x": RandomBot(), "y": RandomBot()}, lineups=lineups,
                  num_envs=2, seed=0)
    assert play_lineups(**kwargs, should_stop=lambda: True) == []
    assert len(play_lineups(**kwargs, should_stop=lambda: False)) == 4


def test_a_failing_pulse_state_write_is_logged_and_training_goes_on(tmp_path, monkeypatch, caplog):
    from colosseum.launcher import Launcher
    from colosseum.metrics import jsonl
    from game_helpers import make_test_run_dir

    cfg = _cfg()
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path))
    launcher._pulse_feeder = PulseFeeder(["main"], every=1, max_backlog=4)
    launcher._pulse_requests = queue.Queue(maxsize=1)

    def broken(path, data):
        raise OSError("disk full on purpose")

    monkeypatch.setattr(jsonl, "write_json_atomic", broken)
    launcher._pulse_feeder.offer("main", "ckpt_v1", 1, 10, {}, final=False)
    with caplog.at_level(logging.ERROR, logger="colosseum.launcher"):
        launcher._pump_pulse()                                             # no exception
        launcher._pulse_feeder.offer("main", "ckpt_v2", 2, 20, {}, final=False)
        launcher._pump_pulse()
    errors = [r for r in caplog.records if r.levelno == logging.ERROR and "disk full on purpose" in r.getMessage()]
    assert len(errors) == 2 and errors[0].exc_info and not errors[1].exc_info   # traceback only the first time
    assert not launcher._pulse_feeder.changed and launcher._pulse_requests.get_nowait().version == 1


class _StopAfter:
    """A stop event that reads as set from its ``n``-th check on (the pulse is then mid-play)."""

    def __init__(self, n: int) -> None:
        self.n, self.calls = n, 0

    def is_set(self) -> bool:
        self.calls += 1
        return self.calls >= self.n


def test_a_stop_during_play_ends_the_pulse_without_sending_the_snapshot(monkeypatch):
    cfg = _cfg(blocks=50, num_envs=2)
    stop = _StopAfter(20)                    # past the request loop's checks, inside play_lineups' step loop
    got = _run_pulse_in_process(cfg, [PulseRequest("main", "ckpt_v1", 1, 10, _numpy_state(cfg, "main"))],
                                monkeypatch, stop=stop)
    assert got == [] and 20 <= stop.calls < 200


def test_one_team_games_need_no_pulse_field_and_two_team_games_still_do(monkeypatch):
    """B-I1 (spec block 6: a score layout's pulse games are the homogeneous team, the field adds cross-play): a
    solo game without scripted / frozen agents plans, prints and plays its pulse with an empty field; a two-team
    game's empty field stays the same ConfigError."""
    cfg = make_test_config("solo", agents={"main": {}}, evaluation={"enabled": True, "blocks": 2, "num_envs": 2})
    (line,) = [x for x in validate_config(cfg).lines if x.startswith("pulse:")]
    assert "field []" in line and "2 matches per snapshot of 'main'" in line
    (message,) = _run_pulse_in_process(cfg, [PulseRequest("main", "ckpt_v3", 3, 30, _numpy_state(cfg, "main"))],
                                       monkeypatch)
    assert {r.uid for r in message.results} == {f"pulse-0123456789ab/P/solo/main@v3/-/{k}/r0" for k in (0, 1)}
    assert message.complete and {s.agent_id for r in message.results for s in r.seats} == {"main"}
    two = make_test_config("turns", agents={"main": {}}, evaluation={"enabled": True})
    with pytest.raises(ConfigError, match="evaluation.field is empty: the config has no scripted / frozen agents"):
        pulse_plan(two, env_spec(two), resolve_player_roles(two, env_spec(two)))


def test_the_texts_about_the_final_snapshot_say_a_local_run_never_evaluates_it(tmp_path, caplog):
    """B-M4 (T6.1 ruling): the final snapshot is offered, but a local run stops before it is played; the validate
    line and the schema say so, and the end-of-run INFO gives the real reason (the stop, or the backlog: T6.1
    deferred)."""
    from colosseum.core.config import EvaluationConfig
    from colosseum.launcher import Launcher
    from game_helpers import make_test_run_dir

    (line,) = [x for x in validate_config(_cfg()).lines if x.startswith("pulse:")]
    assert "the final one always" not in line and "evaluate it with colosseum tournament" in line
    every = EvaluationConfig.model_fields["every"].description
    assert "always" not in every and "colosseum tournament" in every
    cfg = _cfg()
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path))
    launcher._pulse_feeder = feeder = PulseFeeder(["main", "b"], every=1, max_backlog=1)
    feeder.offer("main", "ckpt_v5", 5, 50, {}, final=True)
    feeder.offer("b", "ckpt_v7", 7, 70, {}, final=True)                     # pushes main's final out
    with caplog.at_level(logging.INFO, logger="colosseum.launcher"):
        launcher._finish_pulse()
    messages = sorted(r.getMessage() for r in caplog.records if "final snapshot" in r.getMessage())
    assert messages == [
        "pulse: the final snapshot ckpt_v5 of main was not evaluated (the backlog skipped it); evaluate it with "
        "colosseum tournament",
        "pulse: the final snapshot ckpt_v7 of b was not evaluated (the run stopped first); evaluate it with "
        "colosseum tournament"]
