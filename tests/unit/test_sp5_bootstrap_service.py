"""What a hub serves to joining machines (SP5 T6.2, spec 2.3-2.4): the BootstrapService built from a project."""
from __future__ import annotations

import os
import queue

import grpc
import pytest

import join_kit
from colosseum.bootstrap import installer as I
from colosseum.bootstrap import loader as L
from colosseum.core.run_dir import RunDir
from colosseum.hub import server as hub_server
from colosseum.net.messages import PROTOCOL_VERSION
from colosseum.net.security import HubIdentity


def test_service_from_a_project(tmp_path):
    """The run dir inside the project is left out; old hub archives are pruned (newest 5 kept); the archive
    bytes are held in memory; Info carries the service's fields."""
    project = tmp_path / "proj"
    (project / "examples").mkdir(parents=True)
    (project / "examples" / "game.py").write_text("X = 1\n")
    (project / "experiments" / "r1").mkdir(parents=True)
    (project / "experiments" / "r1" / "metrics.jsonl").write_text("{}")
    out = tmp_path / "hub-bundles"
    out.mkdir()
    for i in range(7):
        (out / f"{i:064x}.tar.gz").write_bytes(b"old")
        os.utime(out / f"{i:064x}.tar.gz", (i, i))
    s = I.make_bootstrap_service(project_dir=project, include=[], exclude=[], max_file_mb=10, requirements=None,
                                 run_dir=project / "experiments", insecure=False, out_dir=out,
                                 protocol=PROTOCOL_VERSION)
    assert not any(f.startswith("project/experiments") for f in s.bundle.files)
    assert "project/examples/game.py" in s.bundle.files
    assert s.archive == s.bundle.path.read_bytes() and s.python.count(".") == 1
    assert s.torch_build == "cpu" or s.torch_build.startswith(("cu", "rocm"))
    assert len(list(out.glob("*.tar.gz"))) == 5 and s.bundle.path.exists()
    assert s.script.startswith(b"#!/bin/sh\n")
    info = L.decode_message(s.info().SerializeToString(), L.INFO_FIELDS)
    assert (info["code_hash"], info["bundle_size"], info["python"]) == (s.bundle.code_hash, s.bundle.size, s.python)
    assert info["requirements"] == list(s.requirements) and len(info["requirements_hash"]) == 64
    assert info["protocol_major"] == PROTOCOL_VERSION[0]


def test_tiny_service_is_what_the_kit_serves(tmp_path):
    s = join_kit.tiny_service(tmp_path, insecure=True)
    assert s.requirements == join_kit.REQUIREMENTS and b"--insecure " in s.script


def test_service_leaves_out_the_identity_cache_and_shm_dirs(tmp_path, monkeypatch):
    """P-PF4-24: the hub identity (CA key, token secret), the colosseum cache and the shared-memory dir are never
    shipped when they lie inside the project (the hub's cwd)."""
    project = tmp_path / "proj"
    (project / "examples").mkdir(parents=True)
    (project / "examples" / "game.py").write_text("X = 1\n")
    for var, sub in (("XDG_DATA_HOME", "xdg-data"), ("XDG_CACHE_HOME", "xdg-cache"), ("COLOSSEUM_SHM_DIR", "shm")):
        monkeypatch.setenv(var, str(project / sub))
    identity = HubIdentity.load_or_create()
    assert identity.path.resolve().is_relative_to(project.resolve())
    (project / "xdg-data" / "notes.txt").write_text("kept")
    (project / "xdg-data" / "colosseum" / "node" / ("f" * 64)).mkdir(parents=True)        # a node identity too
    (project / "xdg-data" / "colosseum" / "node" / ("f" * 64) / "node.key").write_text("secret")
    (project / "xdg-cache" / "colosseum" / "bundles" / "x").mkdir(parents=True)
    (project / "xdg-cache" / "colosseum" / "bundles" / "x" / "f.py").write_text("")
    (project / "xdg-cache" / "other.txt").write_text("kept")
    (project / "shm").mkdir()
    (project / "shm" / "colosseum-seg").write_bytes(b"\0" * 8)
    s = I.make_bootstrap_service(project_dir=project, include=[], exclude=[], max_file_mb=10, requirements=None,
                                 run_dir=project / "runs" / "r1", insecure=False,
                                 out_dir=project / "xdg-cache" / "colosseum" / "hub-bundles", protocol=PROTOCOL_VERSION)
    files = [f for f in s.bundle.files if f.startswith("project/")]
    assert "project/xdg-data/notes.txt" in files and "project/xdg-cache/other.txt" in files
    assert not [f for f in files if f.startswith(("project/xdg-data/colosseum/", "project/xdg-cache/colosseum/",
                                                  "project/shm/"))], files


def test_build_server_serves_the_bootstrap_service(tmp_path):
    """``build_server(bootstrap=)`` registers Info, Bundle and the raw Script (R2) next to the Hub service."""
    service = join_kit.tiny_service(tmp_path, insecure=True)
    identity = HubIdentity.load_or_create(tmp_path / "hub-identity")
    server, ports = hub_server.build_server(inbox=queue.SimpleQueue(), identity=identity, insecure=True,
                                            listen=["127.0.0.1:0"], loopback_plain=False,
                                            blobs=hub_server.BlobStore(RunDir(tmp_path / "run")), max_message_mb=4,
                                            bootstrap=service)
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{ports['127.0.0.1:0']}") as channel:
            assert L.fetch_info(channel, insecure=True)["code_hash"] == service.bundle.code_hash
            script = channel.unary_unary(I.SCRIPT_PATH, request_serializer=None, response_deserializer=None)
            assert script(b"", timeout=10) == service.script
            bundle = channel.unary_stream(f"/{hub_server.BOOTSTRAP_SERVICE}/Bundle")
            parts = list(bundle(L.encode_bundle_request(service.bundle.code_hash),
                                metadata=((L.MD_TOKEN, identity.token.secret),), timeout=10))
            assert b"".join(L.decode_message(p, L.PART_FIELDS)["data"] for p in parts) == service.archive
            with pytest.raises(grpc.RpcError) as e:
                list(bundle(L.encode_bundle_request("0" * 64), metadata=((L.MD_TOKEN, identity.token.secret),),
                            timeout=10))
            assert e.value.code() == grpc.StatusCode.NOT_FOUND
    finally:
        server.stop(None)
