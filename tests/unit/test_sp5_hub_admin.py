"""Admin clients find their hub (SP5 T2.5; spec 3.3, amendment A5): ``--hub ADDR`` (+ token) > ``--run RUN``
(``hub.json``) > the single running hub of this machine; several running hubs are an error that lists them.
(``test_admin_status_answers_on_a_running_hub`` needs the hub process and comes with it, T2.5 (a2).)"""
from __future__ import annotations

import os

import pytest

from colosseum.core.run_state import process_start_ticks
from colosseum.hub import admin
from colosseum.hub.admin import ADMIN_COMMANDS, call, resolve_hub, stream
from colosseum.hub.registry import HubInfo, register_running, this_process_info, write_hub_json
from colosseum.net.security import HubIdentity, SecurityError, hub_identity_dir
from hub_kit import serve


def _info(identity, run_dir, **changes):
    base = dict(run_id="R", run_dir=str(run_dir), addresses=(), loopback="127.0.0.1:41234", tls=False,
                ca_fingerprint=identity.fingerprint, identity_dir=str(identity.path), mode="train")
    return this_process_info(**{**base, **changes})


def test_resolve_hub_prefers_the_flag_then_the_run_then_the_single_running_hub(tmp_path, monkeypatch):
    identity = HubIdentity.load_or_create(hub_identity_dir())          # under the redirected data home
    fetched = []

    def fake_fetch(host, port, *, fingerprint, timeout):
        fetched.append((host, port, fingerprint))
        return b"PEM"

    monkeypatch.setattr(admin, "fetch_hub_ca", fake_fetch)
    other = HubIdentity.load_or_create(tmp_path / "other-hub")

    # --hub ADDR with a token: that address, TLS, the CA fetched and pinned by the token's fingerprint
    target = resolve_hub(hub="10.0.0.5:47100", run=None, token=str(other.token), token_file=None)
    assert (target.address, target.tls, target.ca_pem, target.token) == ("10.0.0.5:47100", True, b"PEM", other.token)
    assert fetched == [("10.0.0.5", 47100, other.fingerprint)]
    # --hub ADDR without a token on the hub machine: the local identity's token; --insecure: plain, no CA
    monkeypatch.delenv("COLOSSEUM_TOKEN", raising=False)
    target = resolve_hub(hub="127.0.0.1:47100", run=None, token=None, token_file=None, insecure=True)
    assert (target.tls, target.ca_pem, target.token) == (False, None, identity.token)

    # --run RUN: hub.json's loopback, plain for a train hub; the token and CA of its identity
    run = tmp_path / "runs" / "x"
    run.mkdir(parents=True)
    write_hub_json(run, _info(identity, run))
    target = resolve_hub(hub=None, run=str(run), token=None, token_file=None)
    assert (target.address, target.tls, target.ca_pem, target.token) == ("127.0.0.1:41234", False, None,
                                                                         identity.token)
    hub_run = tmp_path / "runs" / "h"
    hub_run.mkdir()
    write_hub_json(hub_run, _info(identity, hub_run, mode="hub", tls=True, addresses=("0.0.0.0:47100",),
                                  loopback="127.0.0.1:47100"))
    target = resolve_hub(hub=None, run=str(hub_run), token=None, token_file=None)
    assert (target.address, target.tls, target.ca_pem) == ("127.0.0.1:47100", True, identity.ca_pem)
    with pytest.raises(SecurityError, match="has no hub.json"):
        resolve_hub(hub=None, run=str(tmp_path / "runs"), token=None, token_file=None)

    # neither: the single running hub of this machine; none: an error; two: an error listing both run dirs
    with pytest.raises(SecurityError, match="no running hub"):
        resolve_hub(hub=None, run=None, token=None, token_file=None)
    register_running(identity.path, _info(identity, run))
    target = resolve_hub(hub=None, run=None, token=None, token_file=None)
    assert (target.address, target.tls, target.token) == ("127.0.0.1:41234", False, identity.token)
    parent = os.getppid()
    register_running(identity.path, HubInfo(**{**_info(identity, hub_run).__dict__, "pid": parent,
                                               "proc_start": process_start_ticks(parent)}))
    with pytest.raises(SecurityError) as error:
        resolve_hub(hub=None, run=None, token=None, token_file=None)
    assert str(run) in str(error.value) and str(hub_run) in str(error.value)
    assert fetched == [("10.0.0.5", 47100, other.fingerprint)]           # never fetched for a local hub


def test_admin_commands_include_the_status_command():
    assert ADMIN_COMMANDS[0] == "hub.status" and len(set(ADMIN_COMMANDS)) == len(ADMIN_COMMANDS)


@pytest.mark.parametrize("insecure", [False, True])
def test_call_and_stream_reach_the_hubs_main_loop(tmp_path, insecure):
    """Over a real hub server (``hub_kit.serve``: its ``MiniHubLoop`` answers ``ok <command>`` with the args, and a
    stream with one progress reply before the done one); TLS with the CA fetched for the token, or plain."""
    hub = serve(tmp_path, insecure=insecure)
    try:
        target = resolve_hub(hub=hub.address, run=None, token=str(hub.identity.token), token_file=None,
                             insecure=insecure)
        assert target.tls is not insecure and (target.ca_pem == hub.identity.ca_pem) is not insecure
        assert call(target, "hub.status", {"x": 1}) == {"code": 0, "text": "ok hub.status", "data": {"x": 1}}
        assert list(stream(target, "tournament.submit", {"name": "t"})) == [
            {"code": 0, "text": "progress", "data": None, "done": False},
            {"code": 0, "text": "ok tournament.submit", "data": {"name": "t"}, "done": True}]
    finally:
        hub.stop()


def test_an_insecure_hub_gets_the_local_token_only_on_loopback_or_as_a_running_local_hub(tmp_path, monkeypatch):
    """Review round 1: without TLS nothing proves who listens on ADDR, so the machine's owner token is sent in clear
    only to a loopback address or to an address of a hub running here with this identity."""
    monkeypatch.delenv("COLOSSEUM_TOKEN", raising=False)
    identity = HubIdentity.load_or_create(hub_identity_dir())
    for loopback in ("127.0.0.1:47100", "localhost:47100", "[::1]:47100"):
        assert resolve_hub(hub=loopback, insecure=True).token == identity.token
    with pytest.raises(SecurityError, match="no token for the insecure hub at 10.1.2.3:47100"):
        resolve_hub(hub="10.1.2.3:47100", insecure=True)
    run = tmp_path / "run"
    register_running(identity.path, _info(identity, run, mode="hub", addresses=("10.1.2.3:47100",), loopback=None))
    assert resolve_hub(hub="10.1.2.3:47100", insecure=True).token == identity.token
    register_running(identity.path, _info(identity, run, run_id="S", mode="hub", ca_fingerprint="0" * 64,
                                          addresses=("10.9.9.9:47100",), loopback=None))
    with pytest.raises(SecurityError, match="no token for the insecure hub"):        # another identity's hub
        resolve_hub(hub="10.9.9.9:47100", insecure=True)
    assert resolve_hub(hub="10.9.9.9:47100", token=str(identity.token), insecure=True).token == identity.token


def test_resolve_hub_errors_name_what_is_missing(tmp_path, monkeypatch):
    """No local identity and no token; a malformed address; a hub.json whose identity is not here (a given token
    then works with a fetched CA) or was rotated since; an unreadable identity directory."""
    monkeypatch.delenv("COLOSSEUM_TOKEN", raising=False)
    monkeypatch.setattr(admin, "fetch_hub_ca", lambda host, port, *, fingerprint, timeout: b"FETCHED")
    with pytest.raises(SecurityError, match="no token"):
        resolve_hub(hub="10.0.0.5:47100")
    with pytest.raises(SecurityError, match="malformed hub address"):
        resolve_hub(hub="10.0.0.5", token=str(HubIdentity.load_or_create(tmp_path / "x").token))
    elsewhere = HubIdentity.load_or_create(tmp_path / "elsewhere")
    run = tmp_path / "run"
    run.mkdir()
    write_hub_json(run, HubInfo(**{**_info(elsewhere, run).__dict__, "identity_dir": str(tmp_path / "gone"),
                                   "mode": "hub", "tls": True}))
    with pytest.raises(SecurityError, match="is not on this machine"):
        resolve_hub(run=str(run))
    target = resolve_hub(run=str(run), token=str(elsewhere.token))
    assert (target.address, target.tls, target.ca_pem, target.token) == ("127.0.0.1:41234", True, b"FETCHED",
                                                                         elsewhere.token)
    write_hub_json(run, _info(elsewhere, run, ca_fingerprint="0" * 64))
    with pytest.raises(SecurityError, match="rotated"):
        resolve_hub(run=str(run))

    def unreadable(path=None):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(HubIdentity, "load_or_create", unreadable)
    with pytest.raises(SecurityError, match=f"cannot read the hub identity {elsewhere.path}"):
        resolve_hub(run=str(run))


def test_leaving_a_stream_early_closes_its_channel(tmp_path, monkeypatch):
    closed = []

    class RecordingChannel:
        def __init__(self, channel):
            self._channel = channel

        def __getattr__(self, name):
            return getattr(self._channel, name)

        def close(self):
            closed.append(True)
            self._channel.close()

    real = admin._channel
    monkeypatch.setattr(admin, "_channel", lambda target: RecordingChannel(real(target)))
    hub = serve(tmp_path, insecure=True)
    try:
        replies = stream(resolve_hub(hub=hub.address, token=str(hub.identity.token), insecure=True), "x.y")
        assert next(replies)["text"] == "progress" and not closed
        replies.close()
        assert closed == [True]
    finally:
        hub.stop()
