"""Code bundle (SP5 T6.1, spec 2.3): contents, exclusions, size limit, include / exclude, import roots inside and
outside the project, editable distributions, determinism of the code hash, safe and atomic unpacking."""
from __future__ import annotations

import ast
import gzip
import io
import json
import logging
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from colosseum.bootstrap import bundle as B


def _tree(root: Path, files: dict[str, bytes | str]) -> Path:
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data.encode() if isinstance(data, str) else data)
    return root


def _project(tmp_path: Path) -> Path:
    return _tree(tmp_path / "proj", {
        "examples/game.py": "X = 1\n",
        "bots/data/table.json": "{}",
        "README.md": "hi",
        "runs/r1/metrics.jsonl": "{}",
        "runs/bc/bc.pt": b"\0" * 10,
        "wandb/run-1/files/output.log": "",
        ".git/HEAD": "ref",
        ".venv/lib/x.py": "",
        ".superpowers/notes.md": "",
        "examples/__pycache__/game.cpython-312.pyc": b"\0",
        "examples/old.pyc": b"\0",
        "nested/runs/keep.txt": "nested runs/ is not the top-level runs/",
        "nested/.git/config": "",
        "big.bin": b"\0" * (2 * B.MIB),
    })


def _build(project: Path, tmp_path: Path, **kw):
    kw.setdefault("include", [])
    kw.setdefault("exclude", [])
    kw.setdefault("max_file_mb", 1.0)
    kw.setdefault("roots", [])
    return B.build_bundle(project, out_dir=tmp_path / "out", **kw)


def test_project_files_and_default_exclusions(tmp_path, caplog):
    project = _project(tmp_path)
    with caplog.at_level(logging.WARNING):
        b = _build(project, tmp_path)
    assert set(b.files) == {"project/examples/game.py", "project/bots/data/table.json", "project/README.md",
                            "project/nested/runs/keep.txt"}
    assert b.skipped == ("big.bin",)
    assert "big.bin" in caplog.text and "distributed.bundle.include" in caplog.text
    assert b.python_path == ("project",)
    assert b.path == tmp_path / "out" / f"{b.code_hash}.tar.gz" and b.size == b.path.stat().st_size
    with tarfile.open(b.path) as tar:
        names = tar.getnames()
        assert names[0] == "bundle.json" and names[1:] == sorted(names[1:])
        assert json.loads(tar.extractfile("bundle.json").read()) == {
            "format": 1, "python_path": ["project"], "files": 4}
        for m in tar.getmembers():
            assert m.isreg() and m.mtime == 0 and m.uid == m.gid == 0 and m.uname == m.gname == ""


def test_include_overrides_defaults_and_size_exclude_wins(tmp_path):
    project = _project(tmp_path)
    b = _build(project, tmp_path, include=["runs/bc/*.pt", "big.bin", "*.pyc"], exclude=["bots/data", "README.md"])
    assert "project/runs/bc/bc.pt" in b.files and "project/big.bin" in b.files
    assert "project/runs/r1/metrics.jsonl" not in b.files
    assert "project/examples/old.pyc" in b.files                        # a file pattern beats a file exclusion
    assert not any("__pycache__" in f or "/.git/" in f for f in b.files)  # but opens no excluded directory
    assert "project/bots/data/table.json" not in b.files and "project/README.md" not in b.files
    assert b.skipped == ()


def test_code_hash_is_deterministic_and_follows_content(tmp_path):
    project = _project(tmp_path)
    first = _build(project, tmp_path)
    os.utime(project / "examples" / "game.py", (1, 1))
    (project / "examples" / "game.py").chmod(0o600)
    assert _build(project, tmp_path).code_hash == first.code_hash   # mtime and non-exec mode bits do not count
    (project / "examples" / "game.py").write_text("X = 2\n")
    changed = _build(project, tmp_path)
    assert changed.code_hash != first.code_hash
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(
        [f"{first.code_hash}.tar.gz", f"{changed.code_hash}.tar.gz"])  # no temporary files left


def test_symlinked_directory_is_not_followed(tmp_path, caplog):
    project = _project(tmp_path)
    outside = _tree(tmp_path / "elsewhere", {"secret.txt": "no"})
    (project / "linked").symlink_to(outside, target_is_directory=True)
    (project / "examples" / "alias.py").symlink_to(project / "examples" / "game.py")
    with caplog.at_level(logging.WARNING):
        b = _build(project, tmp_path)
    assert not any(f.startswith("project/linked") for f in b.files)
    assert "project/examples/alias.py" in b.files                       # a symlinked file is stored as content
    assert "symlinked" in caplog.text and "linked" in caplog.text


def test_import_roots_inside_and_outside_the_project(tmp_path):
    project = _project(tmp_path)
    _tree(project, {"src/mylib/__init__.py": "A = 1\n", "src/mylib/core.py": ""})
    outside = _tree(tmp_path / "ext", {"extlib/__init__.py": "", "extlib/__pycache__/x.pyc": b"",
                                      "extlib/runs/__init__.py": "", "single.py": "S = 1\n", "unrelated.txt": "x"})
    whole = _tree(tmp_path / "devsrc", {"devpkg/__init__.py": "", ".git/HEAD": "", "devpkg/__pycache__/a.pyc": b"",
                                        "notes.txt": "kept: a whole sys.path dir"})
    roots = [B.ImportRoot(project / "src", ("mylib",), "mylib"),
             B.ImportRoot(outside, ("extlib",), "extlib"),
             B.ImportRoot(outside, ("single.py",), "single"),
             B.ImportRoot(whole, (), "devpkg")]
    b = _build(project, tmp_path, roots=roots)
    assert b.python_path == ("project", "project/src", "pkg/0", "pkg/1", "pkg/2")
    assert "project/src/mylib/core.py" in b.files                       # inside: stored once, as project files
    assert {"pkg/0/extlib/__init__.py", "pkg/0/extlib/runs/__init__.py", "pkg/1/single.py",
            "pkg/2/devpkg/__init__.py", "pkg/2/notes.txt"} <= set(b.files)
    assert not any("unrelated" in f or "__pycache__" in f or "/.git/" in f for f in b.files)


def test_import_root_under_an_excluded_dir_is_copied(tmp_path):
    project = _project(tmp_path)
    _tree(project, {".venv/lib/site-packages/colosseum/__init__.py": ""})
    roots = [B.ImportRoot(project / ".venv/lib/site-packages", ("colosseum",), "colosseum")]
    b = _build(project, tmp_path, roots=roots)
    assert b.python_path == ("project", "pkg/0") and "pkg/0/colosseum/__init__.py" in b.files


def test_import_roots_finds_colosseum_and_editable_distributions(tmp_path, monkeypatch):
    src = _tree(tmp_path / "devlib" / "src", {"devlib/__init__.py": ""})
    site = tmp_path / "site"
    dist_info = _tree(site, {
        "devlib-1.0.dist-info/METADATA": "Metadata-Version: 2.1\nName: devlib\nVersion: 1.0\n",
        "devlib-1.0.dist-info/direct_url.json": json.dumps(
            {"url": (tmp_path / "devlib").as_uri(), "dir_info": {"editable": True}}),
        "devlib-1.0.dist-info/RECORD": "_devlib.pth,,\ndevlib-1.0.dist-info/METADATA,,\n",
        "_devlib.pth": f"{src}\n",
        "ghost-2.0.dist-info/METADATA": "Metadata-Version: 2.1\nName: ghost\nVersion: 2.0\n",
        "ghost-2.0.dist-info/direct_url.json": json.dumps({"url": "file:///nowhere", "dir_info": {"editable": True}}),
        "ghost-2.0.dist-info/RECORD": "",
        "plain-3.0.dist-info/METADATA": "Metadata-Version: 2.1\nName: plain\nVersion: 3.0\n",
    })
    import importlib.metadata as md
    dists = list(md.distributions(path=[str(dist_info)]))
    roots = B.import_roots(distributions=dists)
    import colosseum
    assert roots[0].path == Path(colosseum.__file__).resolve().parent.parent and roots[0].packages == ("colosseum",)
    assert [r.path for r in roots[1:]] == [src.resolve()] and roots[1].packages == ()


def _archive(tmp_path: Path, members: list[tuple[tarfile.TarInfo, bytes]]) -> tuple[Path, str]:
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        for info, data in members:
            info.size = len(data) if info.isreg() else 0
            tar.addfile(info, io.BytesIO(data) if info.isreg() else None)
    path = tmp_path / "evil.tar.gz"
    path.write_bytes(raw.getvalue())
    return path, B.sha256_file(path)


def _reg(name: str) -> tarfile.TarInfo:
    return tarfile.TarInfo(name)


MANIFEST = (_reg("bundle.json"), json.dumps({"format": 1, "python_path": ["project"], "files": 0}).encode())


@pytest.mark.parametrize("bad", ["/etc/passwd", "../up.txt", "project/../../up.txt", "project//x", "./x", "a\\b"])
def test_unpack_refuses_unsafe_names(tmp_path, bad):
    path, digest = _archive(tmp_path, [MANIFEST, (_reg(bad), b"x")])
    with pytest.raises(B.BundleError, match="unsafe member name"):
        B.unpack_bundle(path, digest, cache_root=tmp_path / "cache")
    assert not (tmp_path / "up.txt").exists()
    assert list((tmp_path / "cache" / "colosseum" / "bundles").iterdir()) == []   # nothing left behind


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE, tarfile.FIFOTYPE])
def test_unpack_refuses_non_regular_members(tmp_path, kind):
    info = _reg("project/link")
    info.type, info.linkname = kind, "/etc/passwd"
    path, digest = _archive(tmp_path, [MANIFEST, (info, b"")])
    with pytest.raises(B.BundleError, match="not a regular file"):
        B.unpack_bundle(path, digest, cache_root=tmp_path / "cache")


def test_unpack_checks_hash_duplicates_manifest_and_format(tmp_path):
    path, digest = _archive(tmp_path, [MANIFEST, (_reg("project/a"), b"1")])
    with pytest.raises(B.BundleError, match="hash mismatch"):
        B.unpack_bundle(path, "0" * 64, cache_root=tmp_path / "cache")
    path, digest = _archive(tmp_path, [MANIFEST, (_reg("project/a"), b"1"), (_reg("project/a"), b"2")])
    with pytest.raises(B.BundleError, match="duplicate"):
        B.unpack_bundle(path, digest, cache_root=tmp_path / "cache")
    path, digest = _archive(tmp_path, [(_reg("project/a"), b"1")])
    with pytest.raises(B.BundleError, match="no bundle.json"):
        B.unpack_bundle(path, digest, cache_root=tmp_path / "cache")
    path, digest = _archive(tmp_path, [(_reg("bundle.json"), b'{"format": 99, "python_path": []}')])
    with pytest.raises(B.BundleError, match="format 99"):
        B.unpack_bundle(path, digest, cache_root=tmp_path / "cache")


def test_unpack_refuses_oversized_content_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "MAX_UNPACKED_BYTES", 10)
    path, digest = _archive(tmp_path, [MANIFEST, (_reg("project/a"), b"x" * 64)])
    with pytest.raises(B.BundleError, match="unpacks to"):
        B.unpack_bundle(path, digest, cache_root=tmp_path / "cache")


def test_round_trip_reuse_and_python_path(tmp_path):
    project = _project(tmp_path)
    outside = _tree(tmp_path / "ext", {"extlib/__init__.py": "VALUE = 7\n"})
    (project / "run.sh").write_text("#!/bin/sh\n")
    (project / "run.sh").chmod(0o755)
    b = _build(project, tmp_path, roots=[B.ImportRoot(outside, ("extlib",), "extlib")])
    cache = tmp_path / "cache"
    target = B.unpack_bundle(b.path, b.code_hash, cache_root=cache)
    assert target == cache / "colosseum" / "bundles" / b.code_hash
    assert (target / "project" / "examples" / "game.py").read_text() == "X = 1\n"
    assert os.access(target / "project" / "run.sh", os.X_OK)
    assert B.bundle_python_path(target) == [target / "project", target / "pkg" / "0"]
    assert B.project_dir(target) == target / "project"
    b.path.unlink()                                                # an unpacked bundle is reused by its hash
    assert B.unpack_bundle(tmp_path / "missing.tar.gz", b.code_hash, cache_root=cache) == target
    out = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path[:0] = sys.argv[1:]; "
                          "import extlib, examples.game; print(extlib.VALUE, examples.game.X)",
                          *map(str, B.bundle_python_path(target))], capture_output=True, text=True,
                         cwd=B.project_dir(target))
    assert out.stdout.split() == ["7", "1"], out.stderr


def test_module_is_standalone():
    """bundle.py is embedded by the installer (T6.2): module-level imports are stdlib only, no colosseum."""
    source = Path(B.__file__).read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                top = name.split(".")[0]
                assert top == "__future__" or top in sys.stdlib_module_names, name
    out = subprocess.run([sys.executable, "-I", "-c", "import importlib.util, sys; "
                          "s = importlib.util.spec_from_file_location('b', sys.argv[1]); m = importlib.util."
                          "module_from_spec(s); sys.modules['b'] = m; s.loader.exec_module(m); "
                          "print('torch' in sys.modules, 'colosseum' in sys.modules)", B.__file__],
                         capture_output=True, text=True)
    assert out.stdout.split() == ["False", "False"], out.stderr


def test_refuses_home_root_and_huge_projects(tmp_path, monkeypatch):
    project = _project(tmp_path)
    monkeypatch.setenv("HOME", str(project))
    with pytest.raises(B.BundleError, match="home or root"):
        _build(project, tmp_path)
    with pytest.raises(B.BundleError, match="home or root"):
        B.build_bundle(Path("/"), include=[], exclude=[], max_file_mb=1, out_dir=tmp_path / "out", roots=[])
    monkeypatch.setenv("HOME", str(tmp_path / "elsewhere"))
    monkeypatch.setattr(B, "MAX_PROJECT_FILES", 3)
    with pytest.raises(B.BundleError, match="distributed.bundle.exclude"):
        _build(project, tmp_path)


def test_relative_pth_line_and_outside_root_containing_the_project(tmp_path):
    parent = _tree(tmp_path / "repo", {"toplib/__init__.py": "", "notes.txt": ""})
    project = _project(parent)                                     # repo/proj: the project inside an editable root
    site = tmp_path / "site"
    _tree(site, {
        "toplib-1.0.dist-info/METADATA": "Metadata-Version: 2.1\nName: toplib\nVersion: 1.0\n",
        "toplib-1.0.dist-info/direct_url.json": json.dumps({"url": parent.as_uri(), "dir_info": {"editable": True}}),
        "toplib-1.0.dist-info/RECORD": "_toplib.pth,,\n",
        "_toplib.pth": "../repo\n",                                 # relative to the .pth file's directory
    })
    import importlib.metadata as md
    roots = B.import_roots(distributions=list(md.distributions(path=[str(site)])))
    assert [r.path for r in roots[1:]] == [parent.resolve()]
    b = _build(project, tmp_path, roots=roots[1:])
    assert "pkg/0/toplib/__init__.py" in b.files and "pkg/0/proj/examples/game.py" in b.files
    assert not any(f.startswith(("pkg/0/proj/runs/", "pkg/0/proj/wandb/")) for f in b.files)


def test_unpack_refuses_bad_manifest_entries_and_missing_archive(tmp_path):
    bad = (_reg("bundle.json"), json.dumps({"format": 1, "python_path": ["/etc"], "files": 0}).encode())
    path, digest = _archive(tmp_path, [bad])
    with pytest.raises(B.BundleError, match="unsafe member name"):
        B.unpack_bundle(path, digest, cache_root=tmp_path / "cache")
    path, digest = _archive(tmp_path, [MANIFEST, (_reg("project/./a"), b"1")])
    with pytest.raises(B.BundleError, match="unsafe member name"):
        B.unpack_bundle(path, digest, cache_root=tmp_path / "cache")
    with pytest.raises(B.BundleError, match="cannot be read"):
        B.unpack_bundle(tmp_path / "missing.tar.gz", "0" * 64, cache_root=tmp_path / "cache")
    with pytest.raises(B.BundleError, match="not a code hash"):
        B.unpack_bundle(path, "../" + digest[3:], cache_root=tmp_path / "cache")
    assert list(B.bundles_dir(tmp_path / "cache").iterdir()) == []


def test_stale_partial_dir_of_the_hash_is_replaced(tmp_path, caplog):
    project = _project(tmp_path)
    b = _build(project, tmp_path)
    stale = B.bundles_dir(tmp_path / "cache") / b.code_hash
    _tree(stale, {"project/half-written.py": ""})                     # no completion marker
    with caplog.at_level(logging.WARNING):
        target = B.unpack_bundle(b.path, b.code_hash, cache_root=tmp_path / "cache")
    assert target == stale and (target / B.COMPLETE_MARKER).is_file()
    assert (target / "project" / "examples" / "game.py").is_file()
    assert not (target / "project" / "half-written.py").exists()
    assert sorted(p.name for p in B.bundles_dir(tmp_path / "cache").iterdir()) == [b.code_hash]
    assert "incomplete directory" in caplog.text


def test_archive_above_the_node_limit_is_refused(tmp_path, monkeypatch):
    project = _project(tmp_path)
    monkeypatch.setattr(B, "MAX_ARCHIVE_BYTES", 64)
    with pytest.raises(B.BundleError, match=r"above the 0 MiB nodes accept.*distributed\.bundle\.exclude"):
        _build(project, tmp_path)
    assert list((tmp_path / "out").iterdir()) == []                  # no archive and no temporary file left
