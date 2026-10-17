# SP5 Plan — Part 6: Joining machines (code bundle, node environment, bootstrap, `join`, registration checks, node admin, WSL2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Scope.** This part owns the following parts of `docs/superpowers/specs/2026-10-17-sp5-distributed-design.md` (Russian; read the spec first, it is the binding authority):
- **Block 2.1** (identity per hub machine, token): the command side — `colosseum hub token [--rotate]`, `--token-file` / `COLOSSEUM_TOKEN` for `join` (T6.2 loader, T6.3 CLI). The library (`HubIdentity`, `Token`, `NodeBook`, `NodeIdentity`, certificates) is T1.3's.
- **Block 2.2 without the SSH mode:** the curl one-liner and the installer script (T6.2); the join command, the hub printing both for each of its addresses, `nodes list`, `nodes remove NAME [--ban]` (T6.3).
- **Block 2.3** (code of a node): the code bundle (T6.1), the thin loader, per-process `PYTHONPATH` on the bundle, the bundle's project dir as working directory (T6.2), the node agent restarting itself through the loader on a new code hash (T6.3).
- **Block 2.4** (node dependencies): the requirement list and the torch build choice (T6.1), the uv environment per (requirements, Python, torch version, build) with the post-install check (T6.2).
- **Block 2.5** (registration checks): T6.3.
- **Block 2.6 on `join` nodes:** the join flags and their slot defaults (T6.3); the hardware probe and `default_slots` are T2.4's, placement T4.1's.
- **Block 3.5** (WSL2 NAT detection and hint): T6.3.
- **Rulings** R1 (application-level node authentication: the loader sends only the token secret, only to `Bootstrap/Bundle`), R2 (`Bootstrap/Script` returns raw bytes), R15 (the hub's TLS certificate is the CA: the loader fetches it unverified and pins its key with a stdlib DER parse).

Not in this part: the SSH mode (`nodes add`, T8.1: it feeds this part's installer script to `sh -s` over SSH), `nodes update-code` (T8.3: it reuses `NodeAgent._reexec_loader`), the hardware probe (T2.4), learner placement (T4.1), hub-away behaviour and exit codes on signals (T4.4), `Dockerfile.node` (T7.3), `docs/DISTRIBUTED_GUIDE.md` (T7.5; this part's messages point to its WSL2 section), the two-machine acceptance (T7.6 runbook).

**What the part delivers.** After T6.3:
- `colosseum hub` and `colosseum train --listen` serve the `Bootstrap` service on the hub port and print, for each address, `colosseum join <addr> --token '...'` and the curl one-liner (the token only on the console, redacted in `main.log`), and the netsh portproxy recipe when the hub runs in WSL2 with NAT networking;
- a machine without colosseum joins with the one-liner: the installer script installs a private uv when none is usable, runs the loader with the hub's Python and grpcio, and the loader pins the hub's key, downloads the code bundle by hash, builds the node environment with uv (the hub's exact versions; `torch==<hub version>` with the build for this machine's driver and GPUs) and execs the node agent from it; `colosseum join ADDR --token ...` does the same from an installed colosseum of any version;
- the hub checks every `Hello` (platform, protocol major, code hash, Python major.minor, torch version, free shared memory and cache disk) and refuses with a reason; a code-hash refusal makes the node fetch the hub's code and restart through the loader; a different torch build is a WARNING;
- nodes persist: they stay after a run, take the next run of the same hub (a new code hash included), keep their number and name; `--once` leaves after the run; `nodes list`, `nodes remove NAME [--ban]`, `hub token [--rotate]` work.

**Read `00-overview.md` first** (global constraints, file map, the binding interface contract). Names used here are the contract's; additions and deviations are listed in `## Contract notes` at the end.

**How a machine joins (the flow; each task builds a piece).**
1. The hub (T6.3 wiring) builds a `BootstrapService` at start (T6.2): the code bundle of its working directory (T6.1 `build_bundle`), the frozen requirement list (`node_requirements`), its Python `major.minor`, torch version and build, and the installer script. `train` without `--listen` builds none.
2. One-liner: `printf '\0\0\0\0\0' | curl ... /colosseum.v1.Bootstrap/Script | tail -c +6 | COLOSSEUM_HUB='<addr>' COLOSSEUM_TOKEN='cst1...' sh`. curl pins the hub's key (`--pinnedpubkey`, the token's fingerprint) over HTTP/2 (`-k`: the pin is the trust); `tail` strips the 5-byte gRPC prefix (R2). The script's body is one function called by its last line, so a cut script runs nothing.
3. The script writes its embedded `loader.py`, `bundle.py`, `torch_select.py` to `<cache>/colosseum/loader/<id>/`, finds uv (`COLOSSEUM_UV`, the private `<cache>/colosseum/uv/uv`, `uv` on PATH) or installs the pinned uv 0.13.1 with `UV_UNMANAGED_INSTALL` (no profile edits, no receipt), and runs `uv run --no-project --python <hub X.Y> --with grpcio==<hub version> -- python -I loader.py --uv <uv> <addr> [flags]`.
4. `colosseum join ADDR [flags]` (T6.3) parses with click and calls `loader_main` in its own process: from here both paths are one.
5. The loader (T6.2): token (`--token` > `--token-file` > `COLOSSEUM_TOKEN`); TLS: `ssl.get_server_certificate` + SPKI SHA-256 = the token's fingerprint, then a gRPC channel with that certificate as root and the fixed name `colosseum-hub` (`--insecure`: plain); `Bootstrap/Info` (open); the bundle by hash (`Bootstrap/Bundle`, token secret in `x-colosseum-token`), verified and unpacked into `<cache>/colosseum/bundles/<hash>/` (T6.1 `unpack_bundle`).
6. The environment (skipped with `COLOSSEUM_NODE_PYTHON`): `torch_select.probe_nvidia` (with `/usr/lib/wsl/lib` on PATH) + `torch_spec` → `--torch-backend` and `UV_CUDA_DRIVER_VERSION`; under an exclusive lock `uv venv --python X.Y <envs>/<key>` + `uv pip install -r <hub list + torch==V> --torch-backend B`; the torch check (version must equal the hub's; GPU present but CPU wheel → WARNING); a marker file last.
7. `exec <env>/bin/python -m colosseum.node.agent <addr> [flags] --code-hash H --bundle DIR`, cwd = the bundle's `project/`, `PYTHONPATH` = the bundle's entries, `PYTHONNOUSERSITE=1`, the caller's `PYTHONPATH` / `PYTHONHOME` / `VIRTUAL_ENV` gone, the token in `COLOSSEUM_TOKEN` (never in argv); shared locks on the bundle's and the environment's `.in-use` files survive the exec, so pruning (newest 5 bundles, newest 2 environments) never removes what a running node uses.
8. The node agent (T2.4; T6.3 adds the argv, the refusal handling and the restart) registers (token → node id, certificate, node secret; T2.3 / T1.3), opens the session with `Hello` (code hash, hardware with torch version / build, free shm and cache bytes); the hub's Session servicer runs `check_hello` (T6.3) and aborts with `FAILED_PRECONDITION` + a JSON `Refusal` before `Welcome`.
9. On a `code_hash` refusal (a new run of a persistent node's hub with other code; T8.3's `update_code` later) the agent stops its children and `exec`s `python -I <bundle>/.../colosseum/bootstrap/loader.py <same join args>` with the token in the environment: back to step 5 with the new hash.

**Conventions used in every task.**
- Run every command from the repository root as `env -u PYTHONPATH .venv/bin/python ...` / `env -u PYTHONPATH .venv/bin/ruff ...` (owner's machine: ROS sets `PYTHONPATH`).
- "Full fast suite" means `env -u PYTHONPATH .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` (zero failures, zero warnings), followed by `env -u PYTHONPATH .venv/bin/ruff check .`.
- New test files start with `test_sp5_` and have unique basenames; new support module `tests/join_kit.py` (T6.2) goes into `known-first-party`.
- Every file a test writes is under `tmp_path`: the autouse fixture `_isolate_colosseum_dirs` (T1.3) points `XDG_DATA_HOME`, `XDG_CACHE_HOME`, `COLOSSEUM_SHM_DIR` into `tmp_path`; a `join` process of a test gets its own three dirs (`cli_runner.node_env`, T6.3). No test installs anything, downloads anything or runs the astral.sh installer: uv is the fake of `join_kit` (`COLOSSEUM_UV`), environments are skipped with `COLOSSEUM_NODE_PYTHON`.
- At most 2 loaded CPU processes per fast test (hub and node agent processes are mostly idle and not counted), at most 3 per slow test; processes through `tests/cli_runner.py`; ports from code that binds port 0 (`--port 0`, loopback servers on `127.0.0.1:0`); waits by counters and files, never by wall-clock values.
- Messages go through `logging`; the join commands with the token go to stdout only (never into a log file). Durations use `time.monotonic()`.
- Commit messages use conventional prefixes and carry no attribution lines; push after every task (`git push`, branch `sp5-distributed` or the task's worktree branch).

**Cross-task notes.**
- **T6.1 is in group A** (parallel worktrees on the unchanged tree): it creates `src/colosseum/bootstrap/{__init__,bundle,requirements,torch_select}.py` and three unit test files, nothing else (no `pyproject.toml` edit, no kit). It imports nothing from other SP5 tasks.
- **Standalone modules.** `bundle.py`, `torch_select.py` and `loader.py` import only the standard library at module level (`loader.py` also `grpc`, inside functions) and nothing from `colosseum`: the installer embeds them verbatim and `colosseum join` runs `loader.py` from installed colosseum versions that may differ from the hub's. Each has a test that checks its module-level imports by AST and loads it by file path under `python -I`. `loader.py` loads its two siblings from the package when it is `colosseum.bootstrap.loader`, else from its own directory (`_sibling`).
- **The loader speaks only `Bootstrap`** and encodes / decodes its three messages (`BootstrapInfo`, `BundleRequest`, `BlobPart`) with a 60-line wire codec, so it needs neither protobuf nor the generated module. Its field-number tables assume T1.2 numbers these messages in the overview's declaration order (1, 2, ...); `test_codec_matches_the_generated_messages` compares the tables with the generated descriptors — if T1.2 numbered differently, change the loader's tables, not the proto. `Bootstrap` is the one service that must stay compatible across versions: add fields, never renumber.
- **T6.2 runs in parallel with T3.x–T5.x:** it creates `bootstrap/installer.py`, `bootstrap/loader.py`, `tests/join_kit.py`, and edits `hub/server.py` only to add `BootstrapServicer`, `add_bootstrap` and the one registration line in `build_server` (`if bootstrap is not None: add_bootstrap(server, bootstrap)`); it does not wire the service into the hub (T6.3 does).
- **T6.3 runs after T5.2** (both edit `cli.py`; T6.3 adds new command functions only) and touches `hub/hub.py` (start-up wiring, registration and admin handlers it names), `hub/server.py` (the Session servicer's `Hello` check), `hub/admin.py` (the `nodes` client calls), `node/agent.py`, `node/session.py` (refusal classification), `node/hardware.py` (`join_slots`), `tests/cli_runner.py`.
- **Source nodes.** `train`'s local node (R12) and the in-process `NodeAgent`s of the T3.x–T5.x cluster tests send `Hello.code_hash = ""`; the hub accepts an empty code hash only from a loopback peer (`registration.check_hello(..., source_node=True)`), so the code-hash check of T6.3 does not break them (proposed ruling 2).
- **Docs:** the WSL2 hint and the loader's messages name `docs/DISTRIBUTED_GUIDE.md (WSL2)`; T7.5 writes that section with the netsh portproxy recipe for Windows 10 first (the owner's machines are Windows 10 build 19045: no mirrored networking, no Hyper-V firewall), then mirrored + Hyper-V rule for Windows 11 22H2+, then Tailscale.

**Verified while writing this part.** Every new module of this part (`bundle.py`, `requirements.py`, `torch_select.py`, `loader.py`, `installer.py`, `registration.py`, `hub/joining.py`, the `BootstrapServicer` / `add_bootstrap` code) and every test file given in full below was run in a scratch package (`/tmp/claude-1000/-home-viv-dev-repos-Colosseum/b41cd787-f9ea-471b-9664-8578a7f7b4b1/scratchpad/part06/`) with the repo's `.venv` (Python 3.12.15, grpcio 1.84.0, cryptography 50.0.2, curl 7.81.0, dash) against stand-ins of the contract names it uses from other tasks (`HubIdentity` with an ECDSA P-256 CA as its TLS certificate, a token-checking `AuthInterceptor` for `Bootstrap`, `send_parts` in 1 MiB parts, a generated `colosseum_v1` proto with the outline's `Bootstrap` messages in declaration order): **108 tests passed with `-W error`, `ruff check` clean** (also the T6.3 unit tests of `registration.py`, `hub/joining.py` and the agent's join functions `join_slots` / `options_from_args` / `join_argv`, against stand-ins of `NodeOptions`, `Hardware` and `default_slots` with the contract's fields and slot rule). In particular:
- the curl one-liner, verbatim from `one_liner`, against a real grpc-python server: TLS with `--pinnedpubkey` (CA as leaf, R15) and `--http2-prior-knowledge` for `--insecure` hubs both run the real installer → the real loader (`uv run` faked) → the recorded agent exec; a wrong pin gives curl exit 90 and runs nothing;
- the script cut at 5 %, 30 %, 50 %, 80 %, 99 %, just before its last line and inside the call runs nothing under `sh` (dash) and bash;
- `ssl.get_server_certificate` + the loader's DER parse give the same fingerprint as `cryptography` (EC and RSA-2048 certificates); the loader's message for each refusal (wrong fingerprint, wrong secret, plain hub / TLS node, TLS hub / `--insecure` node, a hub without `Bootstrap`, nothing listening);
- `build_bundle` on this repository: 667 files, 3.6 MB, 0.4 s, the same code hash on a second build; `unpack_bundle` 0.12 s;
- `node_requirements()` of this `.venv` (50 lines) plus `torch==2.14.1` resolves with `uv pip install --dry-run --torch-backend cpu` (`torch==2.14.1+cpu`) and with `--torch-backend auto` + `UV_CUDA_DRIVER_VERSION=535.54.03` (`nvidia-*-cu12` packages: the cu126 build) (scratch uv cache; nothing installed); a missing package's uv error names it (the loader prints it with the `distributed.requirements` hint);
- `uv run --no-project --python 3.12 --with grpcio==1.84.0 -- python -I -c 'import grpc'` with scratch `UV_CACHE_DIR` / `UV_PYTHON_INSTALL_DIR` downloads a managed Python 3.12.15 and grpcio and runs (11.8 s cold) — the installer's loader step;
- this machine: `probe_nvidia` with a PATH lacking `/usr/lib/wsl/lib` still finds the RTX 4070 Ti (driver 610.88, compute capability 8.9) and `torch_spec("2.14.1", "cu130", ...)` picks `cu130`; `wsl_network()` gives `nat` (`wslinfo`), address `172.31.101.51`; `local_ipv4_addresses()` gives `172.31.101.51`, `192.168.127.2`.
Not verifiable here (no T1.2–T5.2 code yet): the hub / node integration of T6.3 (precise behaviour and test lists below), and a real environment build (no torch download; the acceptance's criterion 8 does it on the laptop).

**Traceability (spec item → task / test).**

| Spec | Item | Task | Test(s) |
|---|---|---|---|
| 2.1 | identity per hub machine, one token for every run | T6.3 | `test_hub_token_prints_the_machine_token` |
| 2.1 | `hub token --rotate`: new CA and secret, every node revoked (refused while a hub of the identity runs: overview deviation) | T6.3 | `test_hub_token_rotate_revokes_and_refuses_while_running` |
| 2.1 | node joins once, keeps its number and name | T6.3 | `test_node_takes_the_next_run_and_new_code` (slow), `test_unknown_node_registers_again_once` |
| 2.1 | `--token-file`, `COLOSSEUM_TOKEN` alternatives to `--token` | T6.2, T6.3 | `test_token_and_address_parsing`, `test_join_cli_token_sources` |
| 2.2 | hub prints `colosseum join <addr>:<port> --token ...` per address | T6.3 | `test_startup_messages`, `test_hub_prints_join_commands_and_serves_bootstrap` |
| 2.2 | one-liner: `Bootstrap/Script` on the hub port, pinned key, `te: trailers`, `tail -c +6`, end marker, uv install, then `join` | T6.2 | `test_one_liner_end_to_end_with_real_curl`, `test_one_liner_with_a_wrong_pin_runs_nothing`, `test_truncated_script_does_nothing`, `test_script_runs_the_loader_and_passes_join_flags`, `test_one_liner_joins_a_real_hub` (slow) |
| 2.2 | `nodes list` (machines, hardware, roles, state, data path, last heartbeat) | T6.3 | `test_nodes_list_shows_online_and_offline_nodes` |
| 2.2 | `nodes remove NAME`: stop the node; `--ban`: id + machine-id refused at re-registration | T6.3 | `test_nodes_remove_releases_and_forgets`, `test_nodes_remove_ban_refuses_registration`, `test_release_stops_the_agent_with_exit_0` |
| 2.3 | thin loader: Bootstrap only, env, `exec` of the node agent; installed colosseum of another version irrelevant | T6.2 | `test_join_builds_env_and_execs_the_agent`, `test_standalone_copy_runs_without_colosseum` |
| 2.3 | agent restarts itself on every code-hash change | T6.3 | `test_code_hash_refusal_reexecs_the_loader`, `test_node_takes_the_next_run_and_new_code` (slow) |
| 2.3 | bundle: package dir, project dir, editable packages outside; exclusions; `max_file_mb`; include / exclude; hash = identity | T6.1 | `test_sp5_code_bundle.py` (all) |
| 2.3 | bundle cache by hash; `PYTHONPATH` per process on the bundle; cwd = project dir | T6.1, T6.2 | `test_round_trip_reuse_and_python_path`, `test_join_builds_env_and_execs_the_agent` |
| 2.3 | weights not in the bundle | T6.1 | (the run dir and `runs/` are excluded: `test_project_files_and_default_exclusions`, `test_service_from_a_project`) |
| 2.4 | exact versions of the hub env without torch / nvidia / triton / colosseum / local paths; `distributed.requirements` | T6.1 | `test_sp5_node_requirements.py` (all) |
| 2.4 | Python `major.minor` of the hub (uv installs it) | T6.2 | `test_join_builds_env_and_execs_the_agent` (`uv venv --python 3.12`) |
| 2.4 | torch of the hub's version, build by hardware (CUDA if the driver supports the hub's build; nearest CUDA build; CPU with WARNING; CPU on Linux without GPU; macOS default wheel) | T6.1, T6.2 | `test_linux_choice`, `test_macos`, `test_gpu_node_with_cpu_torch_warns`, `test_wrong_torch_version_in_env_is_refused` |
| 2.4 | env in `~/.cache/colosseum/envs/<hash>/`; package without a wheel → error naming it + `distributed.requirements` hint | T6.2 | `test_second_join_reuses_bundle_and_env`, `test_failed_install_names_the_package_and_removes_the_env` |
| 2.5 | checks: protocol major, code hash (fetch + restart), Python, torch (refuse) / build (WARNING), platform (native Windows), shm and cache disk space | T6.3 | `test_each_refusal`, `test_accepted_with_build_warning_and_source_nodes`, `test_hello_refusals_reach_the_node`, `test_other_refusals_exit_1_with_the_message` |
| 2.6 | join nodes: workers = physical − 1, evaluators / CPU learners 1 on ≥ 4 physical cores, one learner slot per GPU, `--gpus`, `--cpu-learners`, `--no-learners`, `--workers`, `--evaluators` | T6.3 | `test_join_slots_from_flags` |
| 3.5 | WSL2 NAT detection (`wslinfo`, else `/proc/version` + 172.16/12) and the recipe | T6.3 | `test_wsl_detection`, `test_startup_messages`, `test_hub_logs_the_wsl_hint` |
| 7.5 | `join` exit codes 0 (`--once` after the run, `nodes remove`) / 1 (version mismatch it cannot update from, access refused) / 2 (arguments) | T6.2, T6.3 | `test_refusals`, `test_join_cli_argument_errors_exit_2`, `test_once_exits_after_stop_run`, `test_banned_node_exits_1` |
| R1 / R15 | token secret only in `Bootstrap/Bundle` metadata; hub key pinned before trust | T6.2 | `test_refusals[fingerprint]`, `test_refusals[secret]` |
| R2 | `Bootstrap/Script` raw bytes | T6.2 | `test_one_liner_end_to_end_with_real_curl` |

---

### Task T6.1: `colosseum.bootstrap`: code bundle, node requirements, torch build choice (pure libraries)

Spec 2.3 (archive contents, exclusions, hash, cache by hash), 2.4 (requirement list, torch build by hardware). Three pure modules the hub (T6.2 `make_bootstrap_service`) and the loader (T6.2) use. `bundle.py` and `torch_select.py` are standalone (stdlib only at module level, no colosseum imports): the installer embeds them.

Group A: runs in its own worktree on the unchanged tree, in parallel with T0.1 and T1.1–T1.5. It creates files only.

**Files:**
- Create: `src/colosseum/bootstrap/__init__.py`, `src/colosseum/bootstrap/bundle.py`, `src/colosseum/bootstrap/requirements.py`, `src/colosseum/bootstrap/torch_select.py`
- Test: `tests/unit/test_sp5_code_bundle.py`, `tests/unit/test_sp5_node_requirements.py`, `tests/unit/test_sp5_torch_select.py`

**Interfaces:**
- Consumes: nothing of SP5 (stdlib; `importlib.metadata`; `torch` lazily in `torch_identity`).
- Produces (contract T6.1, with the deviations of `## Contract notes` 1–4):
  - `bundle`: `Bundle(path, code_hash, files, size, python_path=(), skipped=())`, `ImportRoot(path, packages=(), name="")`, `BundleError`, `build_bundle(project_dir, *, include, exclude, max_file_mb, out_dir, roots=None) -> Bundle`, `import_roots(*, distributions=None) -> list[ImportRoot]`, `unpack_bundle(archive, code_hash, *, cache_root) -> Path`, `bundles_dir(cache_root)`, `bundle_python_path(unpacked) -> list[Path]`, `project_dir(unpacked) -> Path`, `sha256_file(path)`; constants `BUNDLE_FORMAT`, `MANIFEST`, `ARCHIVE_SUFFIX`, `COMPLETE_MARKER`, `TOP_EXCLUDES`, `ANY_DIR_EXCLUDES`, `FILE_EXCLUDES`, `MAX_ARCHIVE_BYTES`, `MAX_UNPACKED_BYTES`, `MAX_MEMBERS`, `MAX_PROJECT_FILES`, `MAX_PROJECT_BYTES`;
  - `requirements`: `EXCLUDED`, `canonical_name`, `excluded`, `hub_freeze(distributions=None)`, `read_requirements_file(path)`, `requirements_hash(reqs)`, `node_requirements(*, override, distributions=None) -> (list[str], str)`;
  - `torch_select`: `NvidiaGpu`, `NodePlatform`, `TorchChoice(requirement, backend, cuda_driver, warning)`, `TorchSelectError`, `torch_spec(hub_torch, hub_build, *, platform, gpus) -> TorchChoice`, `probe_platform()`, `probe_nvidia(*, run, environ)`, `nvidia_smi_environ(environ)`, `torch_identity() -> (version, build)`, `build_tag(cuda, hip)`, `public_version`, `parse_version`, `cuda_major`; constants `WSL_LIB`, `MIN_DRIVER`, `CU13_MIN_COMPUTE_CAP`, `OLD_GPU_BACKEND`, `OLD_GPU_LAST_TORCH`.

Rules the code implements (decided here, see `## Contract notes`):
- **Archive layout** `bundle.json` + `project/...` + `pkg/<i>/...`; `bundle.json` = `{"format": 1, "python_path": [...], "files": N}`; member names sorted, regular files only, mtime 0, uid/gid 0, empty owner names, mode 0644 or 0755 (only the exec bit is kept); gzip with mtime 0 and no file name; code hash = SHA-256 of the archive bytes (so the same files give the same hash on every build).
- **Project exclusions:** top-level `runs/`, `.superpowers/`, `wandb/` (WandB writes its run files into the cwd); `.git/`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/` at any depth; `*.pyc`, `*.pyo`; files larger than `max_file_mb` (WARNING listing the first five); symlinked directories are not followed (WARNING); a symlinked file is stored as content. The hub adds its `run.dir` when it lies inside the project (T6.2).
- **`include` / `exclude`** are `fnmatch` patterns on POSIX paths relative to the project, matching a path or any parent directory. `include` overrides the default exclusions and the size limit, but opens a default-excluded directory only when the pattern's literal prefix (up to its first wildcard) names it (`runs/bc/*.pt` does, `*.pt` does not). `exclude` beats everything.
- **Import roots:** the `colosseum` package's `sys.path` entry (from `importlib.util.find_spec`, the package dir only), then each editable distribution (`direct_url.json` `dir_info.editable`) except colosseum: path lines of its `.pth` files (hatchling, flit, pdm, setuptools "compat": the whole directory, with the any-depth exclusions) or the `MAPPING` of a setuptools `__editable___*_finder.py` (the mapped package dirs); a root inside the project (and not excluded) becomes only a `PYTHONPATH` entry `project/<rel>` (the dev checkout's `project/src`), any other root is copied to `pkg/<i>/`. An editable distribution whose sources cannot be found: one WARNING naming `distributed.bundle.include` / `distributed.requirements`.
- **Guards:** a project that is the home directory or `/` is refused; more than `MAX_PROJECT_FILES` (50 000) files or `MAX_PROJECT_BYTES` (1 GiB) to bundle is refused with the `distributed.bundle.exclude` hint (a hub started from the wrong directory must not walk or ship it).
- **Unpacking** (node side; the loader calls it): reuse an unpacked dir that has the completion marker; verify the archive's SHA-256 against the code hash; check every member before writing anything (safe relative names: no absolute path, `..`, empty component, `./` prefix or backslash; regular files only; no duplicates; at most 200 000 members; at most 8 GiB in total; `bundle.json` present with format 1); extract into a temporary sibling dir, write the marker, `os.rename` into `<cache_root>/colosseum/bundles/<hash>/` (losing a race to another loader is fine).
- **Requirements:** `name==version` (PEP 503 names, local version labels such as `+cu130` dropped) of every distribution of the hub environment, sorted, first distribution of a name wins, without `EXCLUDED = ("torch", "nvidia-*", "triton", "colosseum", "cuda-*", "pytorch-triton*")` (on this `.venv` torch 2.14.1 pulls `cuda-bindings`, `cuda-pathfinder`, `cuda-toolkit` besides the `nvidia-*` packages) and without distributions installed from local paths (`direct_url.json` URL `file:`). The hash is the SHA-256 of the joined lines. `distributed.requirements` replaces the list: comments and blanks dropped, pip options refused (`ValueError`), the excluded names dropped with one WARNING.
- **Torch build:** `torch_spec` (rules in the module docstring) returns `torch==<public hub version>` (never a local tag: `+cpu` would pin the CPU wheel even on a GPU node), the uv `--torch-backend` value and the driver version for `UV_CUDA_DRIVER_VERSION`. `probe_nvidia` runs `nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap` with `/usr/lib/wsl/lib` appended to PATH when it exists (uv's own detection misses it in WSL2 from a clean environment and silently installs the CPU wheel), and asks again without `compute_cap` on an old driver.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp5_code_bundle.py`:
```python
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
```

Create `tests/unit/test_sp5_node_requirements.py`:
```python
"""Node requirements (SP5 T6.1, spec 2.4): the hub environment's exact versions minus torch, the CUDA / NVIDIA
packages, colosseum and local-path installs; the distributed.requirements override; the hash."""
from __future__ import annotations

import importlib.metadata as md
import json
import logging
from pathlib import Path

import pytest

from colosseum.bootstrap.requirements import (
    EXCLUDED,
    canonical_name,
    hub_freeze,
    node_requirements,
    read_requirements_file,
    requirements_hash,
)


def _site(tmp_path: Path, dists: dict[str, tuple[str, dict | None]]) -> list[md.Distribution]:
    site = tmp_path / "site"
    for name, (version, direct_url) in dists.items():
        info = site / f"{name}-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        if direct_url is not None:
            (info / "direct_url.json").write_text(json.dumps(direct_url))
    return list(md.distributions(path=[str(site)]))


def test_freeze_excludes_hardware_packages_colosseum_and_local_paths(tmp_path):
    dists = _site(tmp_path, {
        "numpy": ("2.5.4", None), "PyYAML": ("6.0.3", None), "Jinja2": ("3.1.6", None),
        "torch": ("2.14.1", None), "triton": ("3.8.0", None), "nvidia-cublas": ("13.1.1.3", None),
        "nvidia_cudnn_cu13": ("9.24.0.43", None), "cuda-bindings": ("13.4.4", None), "cuda_toolkit": ("13.0.3.0", None),
        "pytorch-triton-rocm": ("3.1.0", None),
        "colosseum": ("0.1.0", {"url": "file:///repo", "dir_info": {"editable": True}}),
        "mylib": ("1.0", {"url": "file:///src/mylib", "dir_info": {"editable": True}}),
        "localwheel": ("2.0", {"url": "file:///tmp/localwheel-2.0.whl", "archive_info": {}}),
        "fromgit": ("0.3", {"url": "https://github.com/x/fromgit", "vcs_info": {"vcs": "git"}}),
        "torchvision": ("0.29.1+cu130", None),
    })
    assert hub_freeze(dists) == ["fromgit==0.3", "jinja2==3.1.6", "numpy==2.5.4", "pyyaml==6.0.3",
                                 "torchvision==0.29.1"]
    assert EXCLUDED == ("torch", "nvidia-*", "triton", "colosseum", "cuda-*", "pytorch-triton*")


def test_first_distribution_of_a_name_wins(tmp_path):
    a = _site(tmp_path / "a", {"numpy": ("2.5.4", None)})
    b = _site(tmp_path / "b", {"numpy": ("1.26.0", None), "torch": ("2.0", None)})
    assert hub_freeze(a + b) == ["numpy==2.5.4"]


def test_override_file(tmp_path, caplog):
    path = tmp_path / "req.txt"
    path.write_text("# mine\nnumpy>=2  # any\n\ntorch==2.0\nnvidia-cublas\nlz4==4.4.5\n")
    with caplog.at_level(logging.WARNING):
        reqs, digest = node_requirements(override=path)
    assert reqs == ["numpy>=2", "lz4==4.4.5"]
    assert digest == requirements_hash(reqs) and len(digest) == 64
    assert "torch==2.0" in caplog.text and "nvidia-cublas" in caplog.text
    path.write_text("-r other.txt\n")
    with pytest.raises(ValueError, match="pip options"):
        read_requirements_file(path)
    path.write_text("<<<\n")
    with pytest.raises(ValueError, match="not a requirement"):
        read_requirements_file(path)


def test_hash_follows_the_list():
    assert requirements_hash(["a==1"]) != requirements_hash(["a==2"])
    assert requirements_hash(["a==1", "b==2"]) != requirements_hash(["b==2", "a==1"])
    assert canonical_name("Jinja2") == "jinja2" and canonical_name("typing_extensions") == "typing-extensions"


def test_real_environment_list_has_no_hardware_packages():
    reqs, _ = node_requirements(override=None)
    names = [r.split("==")[0] for r in reqs]
    assert "numpy" in names and "grpcio" in names
    assert not any(n == "torch" or n.startswith(("nvidia-", "cuda-")) or n in ("triton", "colosseum") for n in names)
    assert all("+" not in r for r in reqs)
```

Create `tests/unit/test_sp5_torch_select.py`:
```python
"""Torch build per node (SP5 T6.1, spec 2.4): the hub's version, the build by the node's hardware (fake probes)."""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from colosseum.bootstrap import torch_select as T
from colosseum.bootstrap.torch_select import NodePlatform, NvidiaGpu, TorchSelectError, torch_spec

LINUX = NodePlatform("Linux", "x86_64")


def gpu(driver="610.88", cc="8.9", name="RTX 4070 Ti"):
    return NvidiaGpu(0, name, 12282, driver, cc)


@pytest.mark.parametrize("hub_build, gpus, backend, driver, warns", [
    ("cu130", [], "cpu", None, False),                                         # Linux without a GPU
    ("cpu", [], "cpu", None, False),
    ("cu130", [gpu()], "cu130", "610.88", False),                               # the hub's build
    ("cu126", [gpu("535.54.03")], "cu126", "535.54.03", False),
    ("cu130", [gpu("535.54.03")], "auto", "535.54.03", True),                   # driver too old for CUDA 13
    ("cpu", [gpu()], "auto", "610.88", False),                                  # a CPU hub, a GPU node
    ("cu130", [gpu("610.88", "6.1", "GTX 1080 Ti")], "cu126", "610.88", True),  # Pascal: CUDA 12.6 build
    ("cu126", [gpu("610.88", "6.1")], "cu126", "610.88", False),
    ("cu130", [gpu("470.1", "6.1")], "cpu", "470.1", True),                     # Pascal + driver below CUDA 12
    ("cu130", [gpu("610.88", None)], "cu130", "610.88", False),                 # no compute_cap reported
    ("cu130", [gpu(cc="8.9"), gpu(cc="7.0")], "cu126", "610.88", True),         # the oldest GPU decides
])
def test_linux_choice(hub_build, gpus, backend, driver, warns):
    choice = torch_spec("2.14.1+cu130", hub_build, platform=LINUX, gpus=gpus)
    assert choice.requirement == "torch==2.14.1"                              # never the local +cuXXX tag
    assert (choice.backend, choice.cuda_driver) == (backend, driver)
    assert (choice.warning is not None) == warns


def test_old_gpu_after_the_last_cuda12_torch_gets_cpu():
    choice = torch_spec("2.15.0", "cu130", platform=LINUX, gpus=[gpu("610.88", "7.0")])
    assert choice.backend == "cpu" and "CPU build" in choice.warning


def test_macos():
    mac = NodePlatform("Darwin", "arm64", "14.7.2")
    assert torch_spec("2.14.1", "cu130", platform=mac, gpus=[]) == T.TorchChoice("torch==2.14.1", None, None, None)
    with pytest.raises(TorchSelectError, match="Intel Macs"):
        torch_spec("2.14.1", "cpu", platform=NodePlatform("Darwin", "x86_64", "13.0"), gpus=[])
    with pytest.raises(TorchSelectError, match="macOS 14"):
        torch_spec("2.14.1", "cpu", platform=NodePlatform("Darwin", "arm64", "13.6"), gpus=[])
    assert torch_spec("2.12.0", "cpu", platform=NodePlatform("Darwin", "arm64", "13.6"), gpus=[]).backend is None
    with pytest.raises(TorchSelectError, match="WSL2"):
        torch_spec("2.14.1", "cpu", platform=NodePlatform("Windows", "AMD64"), gpus=[])


def test_build_tag_and_versions():
    assert T.build_tag("13.0") == "cu130" and T.build_tag("12.6.3") == "cu126"
    assert T.build_tag(None, "7.1.25") == "rocm7.1" and T.build_tag(None) == "cpu"
    assert T.parse_version("2.14.1+cu130") == (2, 14, 1) and T.parse_version("525.60.13") == (525, 60, 13)
    assert T.parse_version("2.15.0rc1") == (2, 15, 0) and T.cuda_major("cu130") == 13 and T.cuda_major("cpu") is None


def _smi(outputs):
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw["env"]["PATH"]))
        rc, out = outputs[len(calls) - 1]
        return subprocess.CompletedProcess(cmd, rc, out, "")
    return run, calls


def test_probe_nvidia_parses_and_retries_without_compute_cap(monkeypatch):
    run, calls = _smi([(0, "0, NVIDIA GeForce RTX 4070 Ti, 12282, 610.88, 8.9\n"
                           "1, Tesla V100, 16384, 535.54.03, 7.0\n")])
    assert T.probe_nvidia(run=run, environ={"PATH": "/usr/bin"}) == [
        NvidiaGpu(0, "NVIDIA GeForce RTX 4070 Ti", 12282, "610.88", "8.9"),
        NvidiaGpu(1, "Tesla V100", 16384, "535.54.03", "7.0")]
    run, calls = _smi([(6, "Field compute_cap is not a valid field"), (0, "0, Old, 4096, 470.1\n")])
    assert T.probe_nvidia(run=run, environ={"PATH": "/usr/bin"}) == [NvidiaGpu(0, "Old", 4096, "470.1", None)]
    assert "compute_cap" in calls[0][0][1] and "compute_cap" not in calls[1][0][1]

    def missing(cmd, **kw):
        raise FileNotFoundError("nvidia-smi")
    assert T.probe_nvidia(run=missing, environ={}) == []


def test_wsl_driver_dir_goes_on_path(monkeypatch):
    monkeypatch.setattr(T.os.path, "isdir", lambda p: p == T.WSL_LIB)
    assert T.nvidia_smi_environ({"PATH": "/usr/bin"})["PATH"] == f"/usr/bin:{T.WSL_LIB}"
    assert T.nvidia_smi_environ({"PATH": f"/usr/bin:{T.WSL_LIB}"})["PATH"] == f"/usr/bin:{T.WSL_LIB}"


def test_module_is_standalone():
    tree = ast.parse(Path(T.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert all(n.split(".")[0] == "__future__" or n.split(".")[0] in sys.stdlib_module_names for n in names)
    out = subprocess.run([sys.executable, "-I", "-c", "import importlib.util, sys; s = importlib.util."
                          "spec_from_file_location('t', sys.argv[1]); m = importlib.util.module_from_spec(s); "
                          "sys.modules['t'] = m; s.loader.exec_module(m); print('torch' in sys.modules)", T.__file__],
                         capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stderr
```

- [ ] **Step 2: Run them (RED)**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_code_bundle.py tests/unit/test_sp5_node_requirements.py tests/unit/test_sp5_torch_select.py -q`
Expected: collection errors `ModuleNotFoundError: No module named 'colosseum.bootstrap'` in all three files.

- [ ] **Step 3: Create the package**

`src/colosseum/bootstrap/__init__.py`:
```python
"""Joining machines (SP5 spec block 2): the code bundle, the node requirements, the torch build choice, the
installer script and the thin ``join`` loader. ``bundle``, ``torch_select`` and ``loader`` are standalone modules
(standard library at module level, no colosseum imports): the installer embeds them."""
```

- [ ] **Step 4: Write `bundle.py`**

`src/colosseum/bootstrap/bundle.py`:
```python
"""The code bundle of a hub (SP5 spec 2.3): one deterministic ``tar.gz`` of the code every node runs.

Archive layout (POSIX member names, sorted, regular files only, zeroed metadata)::

    bundle.json        {"format": 1, "python_path": [...], "files": N}
    project/...        the hub's project directory (its working directory) minus the exclusions
    pkg/<i>/...        import roots that are not part of ``project/``: the ``colosseum`` package when it is
                       installed outside the project, editable packages installed from paths outside it

``python_path`` lists the ``PYTHONPATH`` entries of node processes, relative to the unpacked directory and in the
hub's order: ``project`` first (the hub puts its cwd first on ``sys.path``), then import roots inside the project
(``project/src`` for a source checkout), then ``pkg/<i>``. The code hash is the SHA-256 of the archive bytes.

The module imports only the standard library at module level and nothing from ``colosseum``: the installer
script embeds it next to the loader (T6.2), which unpacks bundles before any colosseum code is on the machine.
"""

from __future__ import annotations

import fnmatch
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import shutil
import stat
import tarfile
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

logger = logging.getLogger(__name__)

BUNDLE_FORMAT = 1
MANIFEST = "bundle.json"
ARCHIVE_SUFFIX = ".tar.gz"
COMPLETE_MARKER = ".colosseum-bundle.json"     # written into the unpacked dir before it is renamed into place
PROJECT = "project"
PKG = "pkg"
MIB = 1 << 20
MAX_ARCHIVE_BYTES = 2 << 30                    # a bundle larger than this is refused (both sides)
MAX_UNPACKED_BYTES = 8 << 30
MAX_MEMBERS = 200_000
# A hub started from the wrong directory (a home dir) must not walk or ship it.
MAX_PROJECT_FILES = 50_000
MAX_PROJECT_BYTES = 1 << 30
# Never bundled from the project (spec 2.3; plus WandB's ./wandb run files): top-level directories, directories at
# any depth, file patterns.
TOP_EXCLUDES = ("runs", ".superpowers", "wandb")
ANY_DIR_EXCLUDES = (".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache")
FILE_EXCLUDES = ("*.pyc", "*.pyo")


class BundleError(Exception):
    """A bundle that cannot be built, verified or unpacked (the message says which file and why)."""


@dataclass(frozen=True)
class ImportRoot:
    """A directory a package is imported from, and what of it goes into the bundle."""

    path: Path                         # the sys.path entry (absolute)
    packages: tuple[str, ...] = ()     # top-level package dirs / module files under it; () = the whole dir
    name: str = ""                     # for messages (distribution name)


@dataclass(frozen=True)
class Bundle:
    path: Path                         # <out_dir>/<code_hash>.tar.gz
    code_hash: str                     # sha256 hex of the archive bytes
    files: tuple[str, ...]             # member names (without bundle.json), sorted
    size: int                          # archive bytes
    python_path: tuple[str, ...] = ()  # PYTHONPATH entries relative to the unpacked dir
    skipped: tuple[str, ...] = ()      # project files left out by the size limit (relative paths)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# import roots of the hub's environment
# ---------------------------------------------------------------------------------------------------------------

def _package_root(name: str) -> ImportRoot | None:
    """The sys.path entry a top-level package or module is imported from (no import of the package itself)."""
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    if spec.submodule_search_locations:
        pkg_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
        return ImportRoot(pkg_dir.parent, (pkg_dir.name,), name)
    if spec.origin and spec.origin not in ("built-in", "frozen"):
        origin = Path(spec.origin).resolve()
        return ImportRoot(origin.parent, (origin.name,), name)
    return None


def _editable_roots(dist: importlib.metadata.Distribution) -> list[ImportRoot] | None:
    """Import roots of an editable distribution: path lines of its ``.pth`` files (hatchling, flit, pdm,
    setuptools "compat"), or the ``MAPPING`` of a setuptools ``__editable___*_finder.py``. None: not found."""
    import ast

    roots: list[ImportRoot] = []
    name = dist.metadata["Name"] or "?"
    for file in dist.files or ():
        located = Path(str(dist.locate_file(file)))
        if file.suffix == ".pth":
            try:
                lines = located.read_text().splitlines()
            except OSError:
                continue
            for line in lines:
                line = line.strip()
                if line and not line.startswith(("#", "import ", "import\t")) and Path(line).is_dir():
                    roots.append(ImportRoot(Path(line).resolve(), (), name))
        elif file.name.startswith("__editable___") and file.name.endswith("_finder.py"):
            try:
                tree = ast.parse(located.read_text())
            except (OSError, SyntaxError):
                continue
            for node in tree.body:
                if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MAPPING"
                                                         for t in node.targets)):
                    try:
                        mapping = ast.literal_eval(node.value)
                    except ValueError:
                        continue
                    for target in mapping.values():
                        p = Path(target).resolve()
                        roots.append(ImportRoot(p.parent, (p.name,), name))
    return roots or None


def import_roots(*, distributions: Iterable[importlib.metadata.Distribution] | None = None) -> list[ImportRoot]:
    """The ``colosseum`` package's root first, then the roots of every other editable distribution of the
    hub's environment (one WARNING per editable distribution whose sources cannot be located)."""
    roots: list[ImportRoot] = []
    own = _package_root("colosseum")
    if own is not None:
        roots.append(own)
    for dist in distributions if distributions is not None else importlib.metadata.distributions():
        raw = dist.read_text("direct_url.json")
        if not raw:
            continue
        try:
            info = json.loads(raw)
        except ValueError:
            continue
        name = (dist.metadata["Name"] or "").lower().replace("_", "-")
        if not info.get("dir_info", {}).get("editable") or name == "colosseum":
            continue
        found = _editable_roots(dist)
        if found is None:
            logger.warning("editable package %s: its sources were not found, so it is not in the code bundle "
                           "(add it with distributed.bundle.include or install it on the nodes with "
                           "distributed.requirements)", dist.metadata["Name"])
            continue
        roots.extend(found)
    return roots


# ---------------------------------------------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------------------------------------------

def _hit(rel: str, patterns: Sequence[str]) -> bool:
    """``rel`` (a POSIX path relative to the project) or one of its parent directories matches a pattern."""
    parts = PurePosixPath(rel).parts
    prefixes = ["/".join(parts[: i + 1]) for i in range(len(parts))]
    return any(fnmatch.fnmatchcase(prefix, pattern.rstrip("/")) for pattern in patterns for prefix in prefixes)


def _may_contain(rel_dir: str, patterns: Sequence[str]) -> bool:
    """An include pattern names something inside the directory ``rel_dir``: its literal prefix (up to the first
    wildcard) is that directory, lies under it or above it. A pattern without a literal prefix (``*.pt``) never
    opens a default-excluded directory."""
    for pattern in patterns:
        literal = pattern.rstrip("/")
        for i, ch in enumerate(literal):
            if ch in "*?[":
                literal = literal[:i]
                break
        literal = literal.rstrip("/")
        if literal and (literal == rel_dir or literal.startswith(rel_dir + "/") or rel_dir.startswith(literal + "/")):
            return True
    return False


def _default_excluded(rel: str, *, is_dir: bool, top_excludes: Sequence[str] = TOP_EXCLUDES) -> bool:
    parts = PurePosixPath(rel).parts
    if parts and parts[0] in top_excludes:
        return True
    dir_parts = parts if is_dir else parts[:-1]
    if any(part in ANY_DIR_EXCLUDES for part in dir_parts):
        return True
    return (not is_dir) and any(fnmatch.fnmatchcase(parts[-1], p) for p in FILE_EXCLUDES)


def _walk(root: Path, *, include: Sequence[str], exclude: Sequence[str], max_bytes: int, skip_dirs: Sequence[Path],
          top_excludes: Sequence[str] = TOP_EXCLUDES) -> tuple[list[tuple[str, Path]], list[str], list[str]]:
    """(rel path, source) of every file of ``root`` that goes into the bundle; skipped big files; skipped
    symlinked directories. ``include`` beats the default exclusions and the size limit; ``exclude`` beats both.
    ``top_excludes``: the top-level directory names left out (the project's ``runs/`` ...; none for import roots)."""
    files: list[tuple[str, Path]] = []
    big: list[str] = []
    links: list[str] = []
    total = 0
    skip = {p.resolve() for p in skip_dirs}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        rel_here = here.relative_to(root).as_posix()
        rel_here = "" if rel_here == "." else rel_here
        kept = []
        for d in sorted(dirnames):
            rel = f"{rel_here}/{d}" if rel_here else d
            full = here / d
            if full.is_symlink():
                links.append(rel)
                continue
            if full.resolve() in skip or _hit(rel, exclude):
                continue
            if _default_excluded(rel, is_dir=True, top_excludes=top_excludes) and not _may_contain(rel, include):
                continue
            kept.append(d)
        dirnames[:] = kept
        for name in sorted(filenames):
            rel = f"{rel_here}/{name}" if rel_here else name
            full = here / name
            if _hit(rel, exclude):
                continue
            included = _hit(rel, include)
            if _default_excluded(rel, is_dir=False, top_excludes=top_excludes) and not included:
                continue
            try:
                st = full.stat()                       # follows a symlinked file
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            if st.st_size > max_bytes and not included:
                big.append(rel)
                continue
            files.append((rel, full))
            total += st.st_size
            if len(files) > MAX_PROJECT_FILES or total > MAX_PROJECT_BYTES:
                raise BundleError(f"{root} holds more than {MAX_PROJECT_FILES} files or {MAX_PROJECT_BYTES >> 20} "
                                  "MiB to bundle: start the hub from the project directory, or leave paths out with "
                                  "distributed.bundle.exclude")
    return files, big, links


def _inside(path: Path, root: Path) -> str | None:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return None


def build_bundle(project_dir: Path, *, include: Sequence[str], exclude: Sequence[str], max_file_mb: float,
                 out_dir: Path, roots: Sequence[ImportRoot] | None = None) -> Bundle:
    """Archive the project dir and the import roots (``import_roots()`` by default) into
    ``out_dir/<code hash>.tar.gz`` (atomic; an existing archive of the same hash is reused)."""
    project = Path(project_dir).resolve()
    if project == Path.home().resolve() or project.parent == project:
        raise BundleError(f"the hub's working directory {project} is a home or root directory: start the hub "
                          "from the project directory (it is what the nodes receive)")
    out = Path(out_dir).resolve()
    roots = import_roots() if roots is None else list(roots)
    max_bytes = int(max_file_mb * MIB)
    files, big, links = _walk(project, include=include, exclude=exclude, max_bytes=max_bytes, skip_dirs=[out])
    members: dict[str, Path] = {f"{PROJECT}/{rel}": src for rel, src in files}
    python_path: list[str] = [PROJECT]
    outside = 0
    for root in roots:
        rel_root = _inside(root.path, project)
        if rel_root not in (None, ".") and not _hit(rel_root, exclude) \
                and not _default_excluded(rel_root, is_dir=True):
            entry = f"{PROJECT}/{rel_root}"           # already in project/: only a PYTHONPATH entry
            if entry not in python_path:
                python_path.append(entry)
            continue
        if rel_root == ".":
            continue                                   # the project dir itself is already entry "project"
        prefix = f"{PKG}/{outside}"
        outside += 1
        python_path.append(prefix)
        if root.packages:                              # package dirs / module files under the root
            for top in root.packages:
                src = root.path / top
                if src.is_dir():
                    sub, sub_big, _ = _walk(src, include=(), exclude=(), max_bytes=max_bytes, skip_dirs=[out],
                                            top_excludes=())
                    big.extend(f"{root.name}:{top}/{r}" for r in sub_big)
                    members.update({f"{prefix}/{top}/{rel}": path for rel, path in sub})
                elif src.is_file():
                    members[f"{prefix}/{top}"] = src
        else:                                          # a whole sys.path directory (an editable .pth entry)
            sub, sub_big, _ = _walk(root.path, include=(), exclude=(), max_bytes=max_bytes, skip_dirs=[out],
                                    top_excludes=())
            big.extend(f"{root.name}:{r}" for r in sub_big)
            members.update({f"{prefix}/{rel}": path for rel, path in sub})
    if big:
        logger.warning("code bundle: %d file(s) larger than distributed.bundle.max_file_mb (%s MB) left out: %s%s "
                       "(add them with distributed.bundle.include)", len(big), max_file_mb, ", ".join(big[:5]),
                       " ..." if len(big) > 5 else "")
    if links:
        logger.warning("code bundle: %d symlinked director%s not followed: %s", len(links),
                       "y" if len(links) == 1 else "ies", ", ".join(links[:5]))
    manifest = json.dumps({"format": BUNDLE_FORMAT, "python_path": python_path, "files": len(members)},
                          sort_keys=True).encode()
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".tmp-{uuid.uuid4().hex}{ARCHIVE_SUFFIX}"
    try:
        with open(tmp, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                    tar.addfile(_info(MANIFEST, len(manifest), 0o644), _BytesReader(manifest))
                    for name in sorted(members):
                        _add_file(tar, name, members[name])
        code_hash = sha256_file(tmp)
        final = out / f"{code_hash}{ARCHIVE_SUFFIX}"
        os.replace(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)
    return Bundle(final, code_hash, tuple(sorted(members)), final.stat().st_size, tuple(python_path), tuple(big))


class _BytesReader:
    def __init__(self, data: bytes) -> None:
        self._data, self._pos = data, 0

    def read(self, n: int = -1) -> bytes:
        end = len(self._data) if n < 0 else self._pos + n
        chunk, self._pos = self._data[self._pos:end], min(end, len(self._data))
        return chunk


def _info(name: str, size: int, mode: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.type = size, 0, tarfile.REGTYPE
    info.mode = 0o755 if mode & 0o111 else 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _add_file(tar: tarfile.TarFile, name: str, src: Path) -> None:
    try:
        with open(src, "rb") as f:
            st = os.fstat(f.fileno())
            tar.addfile(_info(name, st.st_size, st.st_mode), f)
    except OSError as e:
        raise BundleError(f"code bundle: cannot read {src} ({e}); was it changed while the bundle was built?") from e


# ---------------------------------------------------------------------------------------------------------------
# unpacking (node side)
# ---------------------------------------------------------------------------------------------------------------

def bundles_dir(cache_root: Path) -> Path:
    return Path(cache_root) / "colosseum" / "bundles"


def _safe_name(name: str) -> str:
    p = PurePosixPath(name)
    if not name or p.is_absolute() or "\\" in name or any(part in ("..", "") for part in name.split("/")) \
            or name.startswith("./"):
        raise BundleError(f"code bundle: unsafe member name {name!r}")
    return p.as_posix()


def unpack_bundle(archive: Path, code_hash: str, *, cache_root: Path) -> Path:
    """Verify ``archive`` against ``code_hash`` and unpack it into ``<cache_root>/colosseum/bundles/<hash>/``
    (atomic rename; an existing unpacked bundle of that hash is reused). Only regular files with safe relative
    names are accepted; sizes are checked before anything is written."""
    target = bundles_dir(cache_root) / code_hash
    if (target / COMPLETE_MARKER).is_file():
        return target
    archive = Path(archive)
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise BundleError(f"code bundle {archive} is larger than {MAX_ARCHIVE_BYTES >> 20} MiB")
    actual = sha256_file(archive)
    if actual != code_hash:
        raise BundleError(f"code bundle hash mismatch: expected {code_hash}, got {actual}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".tmp-{code_hash[:16]}-{uuid.uuid4().hex[:8]}"
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            members = tar.getmembers()
            if len(members) > MAX_MEMBERS:
                raise BundleError(f"code bundle has {len(members)} members (limit {MAX_MEMBERS})")
            total, names = 0, set()
            for m in members:
                name = _safe_name(m.name)
                if not m.isreg():
                    raise BundleError(f"code bundle: member {m.name!r} is not a regular file")
                if name in names:
                    raise BundleError(f"code bundle: duplicate member {m.name!r}")
                names.add(name)
                total += m.size
            if total > MAX_UNPACKED_BYTES:
                raise BundleError(f"code bundle unpacks to {total} bytes (limit {MAX_UNPACKED_BYTES})")
            if MANIFEST not in names:
                raise BundleError("code bundle has no bundle.json")
            for m in members:
                dest = tmp / _safe_name(m.name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(m)
                assert src is not None
                with src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
                os.chmod(dest, 0o755 if m.mode & 0o111 else 0o644)
        manifest = json.loads((tmp / MANIFEST).read_text())
        if manifest.get("format") != BUNDLE_FORMAT:
            raise BundleError(f"code bundle format {manifest.get('format')} is not supported "
                              f"(this loader reads format {BUNDLE_FORMAT}; update colosseum on this machine)")
        (tmp / COMPLETE_MARKER).write_text(json.dumps({"code_hash": code_hash, "files": len(names)}))
        try:
            os.rename(tmp, target)
        except OSError:
            if not (target / COMPLETE_MARKER).is_file():   # another loader did not win the race: a real error
                raise
    except (tarfile.TarError, OSError, ValueError, EOFError) as e:
        raise BundleError(f"code bundle {archive} cannot be unpacked: {e}") from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return target


def bundle_python_path(unpacked: Path) -> list[Path]:
    """Absolute ``PYTHONPATH`` entries of an unpacked bundle, in order."""
    manifest = json.loads((Path(unpacked) / MANIFEST).read_text())
    return [Path(unpacked) / entry for entry in manifest["python_path"]]


def project_dir(unpacked: Path) -> Path:
    """The working directory of node processes (the bundle's project dir)."""
    return Path(unpacked) / PROJECT
```

- [ ] **Step 5: Write `requirements.py`**

`src/colosseum/bootstrap/requirements.py`:
```python
"""Python requirements of node environments (SP5 spec 2.4): the hub environment's exact versions.

Default list: every installed distribution as ``name==version`` (local version labels such as ``+cu130``
dropped), without the packages whose build depends on the node's hardware (``torch``, ``triton``,
``nvidia-*``, ``cuda-*``, ``pytorch-triton*``), without ``colosseum`` and without distributions installed from
local paths (editable or ``file://``: their sources travel in the code bundle). ``distributed.requirements``
replaces the list with a file of the user's. The loader adds ``torch==<the hub's version>`` itself, with the
build chosen for the node (``torch_select``).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import re
from collections.abc import Iterable
from fnmatch import fnmatchcase
from pathlib import Path

logger = logging.getLogger(__name__)

EXCLUDED = ("torch", "nvidia-*", "triton", "colosseum", "cuda-*", "pytorch-triton*")
_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def canonical_name(name: str) -> str:
    """PEP 503 normalized name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def excluded(name: str) -> bool:
    canon = canonical_name(name)
    return any(fnmatchcase(canon, pattern) for pattern in EXCLUDED)


def _from_local_path(dist: importlib.metadata.Distribution) -> bool:
    raw = dist.read_text("direct_url.json")
    if not raw:
        return False
    try:
        return str(json.loads(raw).get("url", "")).startswith("file:")
    except ValueError:
        return False


def hub_freeze(distributions: Iterable[importlib.metadata.Distribution] | None = None) -> list[str]:
    """``name==version`` of the hub environment's distributions without the exclusions, sorted by name; the first
    distribution of a name wins (``importlib.metadata`` order = ``sys.path`` order)."""
    seen: dict[str, str] = {}
    for dist in distributions if distributions is not None else importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        canon = canonical_name(name)
        if canon in seen or excluded(canon) or _from_local_path(dist):
            seen.setdefault(canon, "")
            continue
        seen[canon] = f"{canon}=={dist.version.split('+', 1)[0]}"
    return sorted(line for line in seen.values() if line)


def read_requirements_file(path: Path) -> list[str]:
    """The lines of a user requirements file: comments and blanks dropped; pip options are refused
    (``ValueError``); torch and the other hardware-dependent packages are dropped with one WARNING."""
    lines: list[str] = []
    dropped: list[str] = []
    for number, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            raise ValueError(f"{path}:{number}: pip options are not supported in distributed.requirements ({line!r})")
        match = _NAME_RE.match(line)
        if match is None:
            raise ValueError(f"{path}:{number}: not a requirement: {line!r}")
        if excluded(match.group(1)):
            dropped.append(line)
            continue
        lines.append(line)
    if dropped:
        logger.warning("distributed.requirements: %s dropped (torch is pinned to the hub's version and its build "
                       "is chosen per node)", ", ".join(dropped))
    return lines


def requirements_hash(requirements: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(requirements) + "\n").encode()).hexdigest()


def node_requirements(*, override: Path | None,
                      distributions: Iterable[importlib.metadata.Distribution] | None = None
                      ) -> tuple[list[str], str]:
    """(requirement lines, their hash): ``override``'s lines, else ``hub_freeze()``."""
    reqs = read_requirements_file(override) if override is not None else hub_freeze(distributions)
    return reqs, requirements_hash(reqs)
```

- [ ] **Step 6: Write `torch_select.py`**

`src/colosseum/bootstrap/torch_select.py`:
```python
"""The torch build of a node (SP5 spec 2.4): the hub's torch version, the build chosen by the node's hardware.

``torch_spec`` returns the requirement (``torch==<hub version>``), the uv ``--torch-backend`` value (``cu130``,
``cpu``, ``auto``; None = the default PyPI wheel, macOS) and the driver version to export as
``UV_CUDA_DRIVER_VERSION`` (with it, uv's choice does not depend on ``PATH``: in WSL2 ``nvidia-smi`` lives in
``/usr/lib/wsl/lib``, which a clean environment lacks, and uv then silently picks the CPU wheel).

Rules (``research/sp5_node_environment.md`` §1):
- macOS: the PyPI wheel (CPU / MPS); torch >= 2.3 has no Intel-Mac wheels, torch >= 2.13 needs macOS 14 on arm64;
- Linux without an NVIDIA GPU: ``cpu``;
- Linux with NVIDIA GPUs: the hub's CUDA build when the driver supports it (CUDA 13 also needs compute
  capability >= 7.5 on every GPU); else, for a GPU older than Turing (< 7.5), ``cu126`` while torch <= 2.14 and the
  driver supports CUDA 12, otherwise ``cpu`` with a warning; else ``auto`` (uv takes the newest build the driver
  supports, with a warning when that is not the hub's build);
- anything else (native Windows): ``TorchSelectError``.

Standard library only at module level and no colosseum imports: the installer embeds this file next to the
loader (T6.2). ``torch_identity`` imports torch lazily (hub and node agent only).
"""

from __future__ import annotations

import os
import platform as _platform
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

WSL_LIB = "/usr/lib/wsl/lib"
CU13_MIN_COMPUTE_CAP = (7, 5)
OLD_GPU_BACKEND = "cu126"
OLD_GPU_LAST_TORCH = (2, 14)            # PyTorch: 2.14 is the last release with CUDA 12 wheels (sm_50..sm_70)
MAC_ARM_ONLY_FROM = (2, 3)
MAC_14_FROM = (2, 13)
# Minimum Linux driver per CUDA major (NVIDIA CUDA Toolkit release notes, table 3; uv LINUX_CUDA_DRIVERS).
MIN_DRIVER = {13: (580, 0, 0), 12: (525, 60, 13), 11: (450, 80, 2)}
_QUERY = "index,name,memory.total,driver_version,compute_cap"


class TorchSelectError(ValueError):
    """The hub's torch cannot be installed on this machine (the message says why and what to do)."""


@dataclass(frozen=True)
class NvidiaGpu:
    index: int
    name: str
    memory_mib: int
    driver: str
    compute_cap: str | None = None       # "8.9"; None: the driver's nvidia-smi does not report it


@dataclass(frozen=True)
class NodePlatform:
    system: str                          # platform.system(): "Linux", "Darwin", "Windows", ...
    machine: str                         # platform.machine(): "x86_64", "arm64", "aarch64"
    mac_release: str | None = None       # platform.mac_ver()[0] on macOS


@dataclass(frozen=True)
class TorchChoice:
    requirement: str                     # "torch==2.14.1"
    backend: str | None                  # uv --torch-backend; None = the default index (macOS)
    cuda_driver: str | None              # export as UV_CUDA_DRIVER_VERSION
    warning: str | None                  # shown by the loader (and logged by the node)


def parse_version(text: str) -> tuple[int, ...]:
    """Leading numeric release components: "2.14.1+cu130" -> (2, 14, 1), "2.15.0rc1" -> (2, 15, 0)."""
    out = []
    for part in text.split("+", 1)[0].split("."):
        digits = ""
        for ch in part:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        out.append(int(digits))
        if digits != part:
            break
    return tuple(out)


def public_version(version: str) -> str:
    return version.split("+", 1)[0]


def build_tag(cuda: str | None, hip: str | None = None) -> str:
    """``torch.version.cuda`` / ``.hip`` -> ``cu130`` / ``rocm7.1`` / ``cpu``."""
    if cuda:
        return "cu" + "".join(cuda.split(".")[:2])
    if hip:
        return "rocm" + ".".join(hip.split(".")[:2])
    return "cpu"


def torch_identity() -> tuple[str, str]:
    """(public version, build tag) of the torch of this interpreter (imports torch)."""
    import torch

    return public_version(torch.__version__), build_tag(torch.version.cuda, getattr(torch.version, "hip", None))


def cuda_major(backend: str) -> int | None:
    if backend.startswith("cu") and backend[2:].isdigit() and len(backend) >= 4:
        return int(backend[2:-1])
    return None


def probe_platform() -> NodePlatform:
    mac = _platform.mac_ver()[0] or None if _platform.system() == "Darwin" else None
    return NodePlatform(_platform.system(), _platform.machine(), mac)


def nvidia_smi_environ(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """``environ`` with WSL2's driver directory on PATH (when it exists)."""
    env = dict(os.environ if environ is None else environ)
    if os.path.isdir(WSL_LIB) and WSL_LIB not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = os.pathsep.join(filter(None, [env.get("PATH"), WSL_LIB]))
    return env


def probe_nvidia(*, run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                 environ: Mapping[str, str] | None = None) -> list[NvidiaGpu]:
    """NVIDIA GPUs from ``nvidia-smi`` (no torch); [] without a driver. An old driver that does not know
    ``compute_cap`` is asked again without it."""
    env = nvidia_smi_environ(environ)
    for query in (_QUERY, _QUERY.rsplit(",", 1)[0]):
        try:
            out = run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                      capture_output=True, text=True, timeout=20, env=env)
        except (OSError, subprocess.SubprocessError):
            return []
        if out.returncode != 0:
            continue
        gpus = []
        for line in out.stdout.strip().splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < 4:
                continue
            try:
                cc = cols[4] if len(cols) > 4 and parse_version(cols[4]) else None
                gpus.append(NvidiaGpu(int(cols[0]), cols[1], int(float(cols[2])), cols[3], cc))
            except ValueError:
                continue
        return gpus
    return []


def torch_spec(hub_torch: str, hub_build: str, *, platform: NodePlatform,
               gpus: Sequence[NvidiaGpu]) -> TorchChoice:
    version = public_version(hub_torch)
    req = f"torch=={version}"
    tv = parse_version(version)
    if platform.system == "Darwin":
        if tv >= MAC_ARM_ONLY_FROM and platform.machine not in ("arm64", "aarch64"):
            raise TorchSelectError(f"torch {version} has no wheels for Intel Macs (the last one is 2.2.2); "
                                   "use an Apple Silicon Mac or another machine")
        if tv >= MAC_14_FROM and platform.mac_release and parse_version(platform.mac_release) < (14,):
            raise TorchSelectError(f"torch {version} needs macOS 14 or newer (this Mac: {platform.mac_release})")
        return TorchChoice(req, None, None, None)
    if platform.system != "Linux":
        raise TorchSelectError(f"{platform.system} is not supported for nodes (Linux, macOS, or Windows via WSL2)")
    if not gpus:
        return TorchChoice(req, "cpu", None, None)
    driver = min((g.driver for g in gpus), key=parse_version)
    dv = parse_version(driver)
    caps = [parse_version(g.compute_cap) for g in gpus if g.compute_cap]
    cc = min(caps) if caps else None
    old_gpu = cc is not None and cc < CU13_MIN_COMPUTE_CAP
    hub_major = cuda_major(hub_build)
    if hub_major is not None and dv >= MIN_DRIVER.get(hub_major, (10 ** 6,)) and not (hub_major >= 13 and old_gpu):
        return TorchChoice(req, hub_build, driver, None)
    gpu_text = f"{gpus[0].name} (driver {driver}" + (f", compute capability {'.'.join(map(str, cc))})" if cc else ")")
    if old_gpu:
        if tv[:2] <= OLD_GPU_LAST_TORCH and dv >= MIN_DRIVER[12]:
            return TorchChoice(req, OLD_GPU_BACKEND, driver,
                               f"{gpu_text} is older than Turing: using the {OLD_GPU_BACKEND} build of torch {version}"
                               + (f" instead of the hub's {hub_build}" if hub_build != OLD_GPU_BACKEND else ""))
        return TorchChoice(req, "cpu", driver,
                           f"no CUDA build of torch {version} supports {gpu_text}: this node uses the CPU build "
                           "(learners on it run on the CPU)")
    note = (f"the driver of {gpu_text} is too old for the hub's {hub_build} build of torch {version}: uv picks the "
            "newest CUDA build this driver supports (a CPU build if there is none; update the driver to avoid it)"
            if hub_major is not None else None)
    return TorchChoice(req, "auto", driver, note)
```

- [ ] **Step 7: Run the new tests (GREEN)**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_code_bundle.py tests/unit/test_sp5_node_requirements.py tests/unit/test_sp5_torch_select.py -q -rw`
Expected: all pass (44 tests, under 2 s), no warnings. `test_real_environment_list_has_no_hardware_packages` reads the real `.venv`: on the owner's CUDA machine it holds `cuda-*` and `nvidia-*` packages, which must all be filtered.

Also check by hand once (report the numbers): `env -u PYTHONPATH .venv/bin/python -c "from pathlib import Path; from colosseum.bootstrap.bundle import build_bundle; import tempfile, time; t = time.monotonic(); b = build_bundle(Path('.'), include=[], exclude=[], max_file_mb=10, out_dir=Path(tempfile.mkdtemp())); print(len(b.files), b.size, b.python_path, round(time.monotonic() - t, 2))"` — expected on this repo: about 670 files, about 3.6 MB, `('project', 'project/src')`, under 1 s.

- [ ] **Step 8: Full fast suite + ruff**

Run the full fast suite and ruff. Expected: green, zero warnings.

- [ ] **Step 9: Commit and push**

```bash
git add src/colosseum/bootstrap tests/unit/test_sp5_code_bundle.py tests/unit/test_sp5_node_requirements.py \
        tests/unit/test_sp5_torch_select.py
git commit -m "feat: code bundle, node requirements and torch build choice for joining machines"
git push
```

---

### Task T6.2: Bootstrap service, installer script, curl one-liner, the `join` loader

Spec 2.2 (one-liner: `Bootstrap/Script` on the hub port, pinned key, end check, uv install, then `join`), 2.3 (thin loader: `Bootstrap` only, bundle by hash, environment, `exec` of the node agent from it, `PYTHONPATH` per process, project cwd), 2.4 (Python `major.minor`, torch build per node, environment per hash, the error naming a package without a wheel), 3.2 / 3.3 for the loader (pinned TLS; `--insecure` plain); rulings R1, R2, R15. Tests run the real loader against a real gRPC server on loopback with a fake uv and a fake environment python (`tests/join_kit.py`); nothing is downloaded or installed.

Runs after T2.5 and T6.1, in parallel with any task from T3.1 to T5.2: it edits `hub/server.py` only to add `BootstrapServicer`, `add_bootstrap` and their registration line in `build_server`. The hub does not build a `BootstrapService` yet (T6.3 wires it in).

**Files:**
- Create: `src/colosseum/bootstrap/loader.py`, `src/colosseum/bootstrap/installer.py`, `tests/join_kit.py`
- Modify: `src/colosseum/hub/server.py` (`BootstrapServicer`, `add_bootstrap`, `build_server`'s `bootstrap=` line), `pyproject.toml` (`join_kit` in `known-first-party`)
- Test: `tests/unit/test_sp5_join_loader.py`, `tests/unit/test_sp5_bootstrap_service.py`, `tests/integration/test_sp5_join_installer.py`

**Interfaces:**
- Consumes: T6.1 `colosseum.bootstrap.{bundle,requirements,torch_select}`; T1.2 `colosseum.net.proto.colosseum_v1_pb2` (`BootstrapInfo`, `BundleRequest`, `BlobPart`), `colosseum_v1_pb2_grpc` (`BootstrapServicer`, `add_BootstrapServicer_to_server`), `colosseum.net.messages.PROTOCOL_VERSION`; T1.3 `colosseum.net.security.{HubIdentity, spki_fingerprint}` (`load_or_create(path)`, `.token` with `.secret`, `.fingerprint`, `.ca_pem`, `.nodes`, `.server_credentials()`), `colosseum.net.grpc_util.{AuthInterceptor, server_options}`; T2.3 `grpc_util.send_parts` (parts with `offset`, `total_size`, `last`), `hub.server.build_server(..., bootstrap=)`.
- Produces (contract T6.2, with the deviations of `## Contract notes` 5–9):
  - `colosseum.bootstrap.loader`: `loader_main(argv, *, environ=None, exec_fn=None) -> int`, `LoaderError(message, code)`, `build_parser()`, `agent_args(ns)`, `agent_environ(environ, bundle_dir, token)`, `resolve_token(token, token_file, environ) -> (token, fingerprint)`, `parse_address(address) -> (host, port)`, `spki_sha256(cert_der)`, `curl_pin(fingerprint)`, `fetch_hub_ca(host, port, fingerprint, *, timeout)`, `open_channel(...)`, `fetch_info(channel, *, insecure)`, `ensure_bundle(channel, info, token, cache)`, `ensure_uv(explicit, environ, cache)`, `ensure_env(info, *, uv, cache, environ, choice, gpus)`, `env_key(info, backend)`, `check_env(python, env)`, `prune(directory, keep, current)`, `decode_message(buf, fields)`, `encode_bundle_request(code_hash)`; constants `INFO_FIELDS`, `PART_FIELDS`, `BUNDLE_REQUEST_CODE_HASH`, `UV_VERSION = "0.13.1"`, `MIN_UV = (0, 7, 14)`, `NODE_PYTHON_ENV = "COLOSSEUM_NODE_PYTHON"`, `UV_ENV = "COLOSSEUM_UV"`, `TOKEN_ENV`, `ENV_MARKER`, `IN_USE`, `KEEP_ENVS = 2`, `KEEP_BUNDLES = 5`, `AGENT_MODULE = "colosseum.node.agent"`, `CLEAN_ENV`;
  - `colosseum.bootstrap.installer`: `installer_script(*, insecure, python, grpcio, uv_version=UV_VERSION) -> bytes`, `one_liner(address, token, *, insecure) -> str`, `join_command(address, token, *, insecure) -> str`, `curl_pin`, `END_MARKER`, `SCRIPT_PATH`, `BootstrapService(bundle, requirements, requirements_hash, python, torch_version, torch_build, script, protocol, archive)` with `.info() -> pb.BootstrapInfo`, `make_bootstrap_service(*, project_dir, include, exclude, max_file_mb, requirements, run_dir, insecure, out_dir, protocol) -> BootstrapService`, `prune_archives(out_dir, *, keep, current)`, `KEEP_HUB_BUNDLES = 5`;
  - `colosseum.hub.server`: `BootstrapServicer(service)`, `add_bootstrap(server, service)`, `BOOTSTRAP_SERVICE`;
  - `tests/join_kit.py`: `write_fakes(directory) -> (uv, python)`, `tiny_service(tmp_path, *, insecure=False, torch_build="cu130", extra_files=None)`, `bootstrap_server(service, identity, *, tls=True) -> (server, port)`, `REQUIREMENTS`, `FAKE_PYTHON`, `FAKE_UV`.
- The agent's command line (consumed by T6.3's `node.agent.main`): `python -m colosseum.node.agent ADDR [--workers N] [--evaluators N] [--cpu-learners N] [--gpus LIST] [--name NAME] [--tags a,b] [--no-learners] [--once] [--insecure] --code-hash H --bundle DIR`, token in `COLOSSEUM_TOKEN`, cwd = `DIR/project`.

Behaviour the code implements (decided here):
- **The loader's environment key** is `sha256(requirements_hash | python | torch_version | backend)[:24]`: one environment per hub requirement list, Python, torch version and build of this machine (`envs/<key>/`; the contract's `envs/<requirements hash>/` does not separate builds).
- **uv lookup:** `COLOSSEUM_UV` > `--uv` (the installer's) > `<cache>/colosseum/uv/uv` > `uv` on PATH; each must report a version ≥ 0.7.14 (`--torch-backend` without `--preview`); none → install the pinned uv 0.13.1 with `UV_UNMANAGED_INSTALL=<cache>/colosseum/uv` (never `UV_INSTALL_DIR`: it rewrites the user's `~/.config/uv/uv-receipt.json`). uv runs with `UV_CACHE_DIR=<cache>/colosseum/uv-cache` and `UV_PYTHON_INSTALL_DIR=<cache>/colosseum/python` unless the user set them (same filesystem as the environments: hard links; the managed Python must live as long as the environments).
- **Environment build** under an exclusive `flock` on `envs/<key>.lock`; the marker `.colosseum-env.json` (the torch check's result) is written last; any failure removes the directory. `uv pip install -r` (not `uv pip sync`: sync installs no dependencies, and torch's `nvidia-*` / `triton` set depends on the build).
- **Torch check** after the install, in the new interpreter (`python -I -c ...`): torch's public version must equal the hub's (else exit 1); an NVIDIA GPU with a CPU wheel, or a CUDA wheel without CUDA, is a WARNING on stderr (spec 2.4 "иначе CPU с WARNING"); the hub sees the build in `Hello` (T6.3).
- **Pruning:** after use the loader keeps the newest 5 bundles and 2 environments (by the mtime of their `.in-use` file, touched on each use) and never removes one another process holds (a shared `flock` on `.in-use` that the node agent inherits through `exec`).
- **Errors** (stderr `colosseum join: ...`, exit 1; 2 for arguments, a missing or malformed token): unreachable hub (with the WSL2 pointer), TLS failure (hint `--insecure`), a key that is not the token's (another hub, a rotated token, an interception), a refused token secret, a hub without `Bootstrap` (plain `train`: start `colosseum hub` or `colosseum train --listen`), a plain connection refused by a TLS hub (drop `--insecure`), `NOT_FOUND` / malformed messages, a bundle of a wrong size or hash, a failed `uv venv` / `uv pip install` (uv's last 12 stderr lines + the `distributed.requirements` hint), a torch version mismatch, `TorchSelectError` (Intel Mac, macOS < 14, native Windows).
- **Installer script** (the module docstring): one function called by the last line `_colosseum_install "$@" # colosseum-installer-end`; the three files in quoted heredocs (`<<'COLOSSEUM_EOF_<12 hex>'`, the tag checked absent from the file); `boot` dir named by a hash of the embedded files; the hub address from `COLOSSEUM_HUB` (exit 2 when unset), the token from `COLOSSEUM_TOKEN` (exit 2), extra arguments passed to the loader (`... | sh -s -- --workers 4`); `uname -s` other than Linux / Darwin → exit 1; `--insecure` baked in when the hub runs insecure. A cut anywhere before the last line runs nothing (the function body is parsed whole); a cut inside the trailing comment runs the complete script, which is correct.
- **One-liner** (`one_liner`): `printf '\0\0\0\0\0' | curl -fsS --http2 -k --pinnedpubkey 'sha256//<b64>' -H 'content-type: application/grpc' -H 'te: trailers' --data-binary @- https://<addr>/colosseum.v1.Bootstrap/Script | tail -c +6 | COLOSSEUM_HUB='<addr>' COLOSSEUM_TOKEN='<token>' sh`; for an `--insecure` hub `--http2-prior-knowledge` and `http://`. The spec's form plus `COLOSSEUM_HUB` (one script serves every address and the SSH mode; contract note 7). `te: trailers` is required by grpc-core; `-L` is never used (a pin does not cover redirects).
- **`BootstrapService`** holds the archive bytes in memory (a few MB): the hub's archive files in `<cache>/colosseum/hub-bundles/` can then be pruned (newest 5 kept) without breaking a running hub. `make_bootstrap_service` excludes the configured `run.dir` when it lies inside the project and is not under `runs/`.

- [ ] **Step 1: Test support `tests/join_kit.py`**

Create `tests/join_kit.py`:
```python
"""Support for the join tests (SP5 T6.2, T6.3): fake ``uv`` and fake environment ``python`` scripts, a tiny
``BootstrapService``, a real Bootstrap gRPC server on loopback (``AuthInterceptor`` + ``add_bootstrap``).

The fakes never touch the network or the machine: ``uv venv DIR`` writes ``DIR/bin/python`` (the fake python);
``uv pip install`` records its arguments and fails with ``FAKE_UV_FAIL``'s text when set; ``uv run ... -- python
ARGS`` executes the real interpreter of the test run with ARGS (so the installer really runs the loader); ``uv
--version`` prints 0.13.1. The fake python answers the loader's torch check with ``FAKE_TORCH_JSON`` (default: a CPU
torch 2.14.1) and otherwise records ``argv`` / ``cwd`` / ``env`` as one JSON line in ``FAKE_RECORD`` and exits 0:
that is the "node agent" the loader execs.
"""
from __future__ import annotations

import stat
import sys
from concurrent import futures
from pathlib import Path

import grpc

from colosseum.bootstrap.bundle import build_bundle
from colosseum.bootstrap.installer import BootstrapService, installer_script
from colosseum.hub.server import add_bootstrap
from colosseum.net.grpc_util import AuthInterceptor, server_options
from colosseum.net.messages import PROTOCOL_VERSION

REQUIREMENTS = ("numpy==2.5.4", "lz4==4.4.5")


FAKE_PYTHON = '''#!{python}
"""Fake env python: the torch check prints FAKE_TORCH_JSON; anything else records argv / cwd / env and exits."""
import json, os, sys
if "-c" in sys.argv:
    print(os.environ.get("FAKE_TORCH_JSON", json.dumps({{"python": "3.12", "torch": "2.14.1+cpu", "cuda": None,
          "cuda_available": False, "mps": False}})))
    raise SystemExit(0)
with open(os.environ["FAKE_RECORD"], "a") as f:
    f.write(json.dumps({{"argv": sys.argv, "cwd": os.getcwd(), "env": dict(os.environ)}}) + "\\n")
'''

FAKE_UV = '''#!{python}
"""Fake uv: --version; venv DIR (a fake python); pip install (records, may fail); run ... -- python ARGS (real)."""
import json, os, sys
args = sys.argv[1:]
log = os.environ.get("FAKE_UV_LOG")
if log:
    with open(log, "a") as f:
        uv_env = {{k: v for k, v in os.environ.items() if k.startswith("UV_")}}
        f.write(json.dumps({{"args": args, "env": uv_env}}) + "\\n")
if args == ["--version"]:
    print("uv 0.13.1 (fake)"); raise SystemExit(0)
if args[0] == "venv":
    target = args[-1]
    os.makedirs(os.path.join(target, "bin"), exist_ok=True)
    py = os.path.join(target, "bin", "python")
    with open(py, "w") as f:
        f.write({fake_python!r})
    os.chmod(py, 0o755)
    raise SystemExit(0)
if args[:2] == ["pip", "install"]:
    fail = os.environ.get("FAKE_UV_FAIL")
    if fail:
        print(fail, file=sys.stderr); raise SystemExit(1)
    raise SystemExit(0)
if args[0] == "run":
    rest = args[args.index("--") + 1:]
    assert rest[0] == "python"
    os.execv({python!r}, [{python!r}, *rest[1:]])
raise SystemExit(f"fake uv: unexpected {{args}}")
'''


def write_fakes(directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    fake_python = FAKE_PYTHON.format(python=sys.executable)
    uv = directory / "uv"
    uv.write_text(FAKE_UV.format(python=sys.executable, fake_python=fake_python))
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    py = directory / "python"
    py.write_text(fake_python)
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    return uv, py


def tiny_service(tmp_path: Path, *, insecure: bool = False, torch_build: str = "cu130",
                 extra_files: dict[str, bytes] | None = None) -> BootstrapService:
    """A BootstrapService over a two-file project (no import roots), the hub's real installer script."""
    project = tmp_path / "hub-project"
    (project / "examples").mkdir(parents=True, exist_ok=True)
    (project / "examples" / "game.py").write_text("X = 1\n")
    for rel, data in (extra_files or {}).items():
        (project / rel).parent.mkdir(parents=True, exist_ok=True)
        (project / rel).write_bytes(data)
    bundle = build_bundle(project, include=[], exclude=[], max_file_mb=10, out_dir=tmp_path / "hub-bundles", roots=[])
    script = installer_script(insecure=insecure, python="3.12", grpcio="1.84.0")
    return BootstrapService(bundle, REQUIREMENTS, "r" * 64, "3.12", "2.14.1", torch_build, script,
                            PROTOCOL_VERSION, bundle.path.read_bytes())


def bootstrap_server(service: BootstrapService | None, identity, *, tls: bool = True) -> tuple[grpc.Server, int]:
    """A started gRPC server on 127.0.0.1:<free port> with the hub's AuthInterceptor and (unless ``service`` is
    None) the Bootstrap service; TLS with the hub identity's certificate, or plain."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=server_options(64),
                         interceptors=[AuthInterceptor(lambda: identity.token.secret, identity.nodes)])
    if service is not None:
        add_bootstrap(server, service)
    if tls:
        port = server.add_secure_port("127.0.0.1:0", identity.server_credentials())
    else:
        port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, port
```
and add `"join_kit"` to `known-first-party` in `pyproject.toml` (sorted list).

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_sp5_join_loader.py`:
```python
"""The join loader (SP5 T6.2, spec 2.3-2.4) against a real gRPC Bootstrap server on loopback: pinned TLS, Info,
the bundle by hash, the uv environment (fake uv), the exec of the node agent (recorded), refusals."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

import join_kit
from colosseum.bootstrap import bundle as B
from colosseum.bootstrap import loader as L
from colosseum.bootstrap import torch_select as T
from colosseum.net.messages import PROTOCOL_VERSION
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.net.security import HubIdentity, spki_fingerprint


@pytest.fixture
def hub(tmp_path):
    identity = HubIdentity.load_or_create(tmp_path / "hub-identity")
    servers = []

    def start(*, tls=True, service="default", **kw):
        svc = join_kit.tiny_service(tmp_path, insecure=not tls, **kw) if service == "default" else service
        server, port = join_kit.bootstrap_server(svc, identity, tls=tls)
        servers.append(server)
        hub.service = svc
        return port
    hub = type("Hub", (), {"identity": identity, "token": str(identity.token), "start": staticmethod(start)})
    yield hub
    for s in servers:
        s.stop(0)


def _env(tmp_path, **extra):
    uv, py = join_kit.write_fakes(tmp_path / "fakes")
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path / "home"), "XDG_CACHE_HOME": str(tmp_path / "cache"),
           "FAKE_RECORD": str(tmp_path / "record.jsonl"), "FAKE_UV_LOG": str(tmp_path / "uv.jsonl"),
           "COLOSSEUM_UV": str(uv), "PYTHONPATH": "/opt/ros/iron/lib/python3.10/site-packages"}
    env.update(extra)
    return env, py


def _run(argv, env, monkeypatch, gpus=()):
    monkeypatch.setattr(T, "probe_nvidia", lambda **kw: list(gpus))
    monkeypatch.setattr(T, "probe_platform", lambda: T.NodePlatform("Linux", "x86_64"))
    calls = []
    code = L.loader_main(argv, environ=env, exec_fn=lambda path, args, e: calls.append((path, args, e)))
    return code, calls


def _uv_calls(tmp_path):
    p = tmp_path / "uv.jsonl"
    return [json.loads(line)["args"] for line in p.read_text().splitlines()] if p.exists() else []


# --- codec and certificate --------------------------------------------------------------------------------------

def test_codec_matches_the_generated_messages():
    msg = pb.BootstrapInfo(protocol_major=1, protocol_minor=3, code_hash="c" * 64, requirements_hash="r" * 64,
                           requirements=["a==1", "b==2"], python="3.12", torch_version="2.14.1",
                           torch_build="cu130", bundle_size=5_000_000_000)
    assert L.decode_message(msg.SerializeToString(), L.INFO_FIELDS) == {
        "protocol_major": 1, "protocol_minor": 3, "code_hash": "c" * 64, "requirements_hash": "r" * 64,
        "requirements": ["a==1", "b==2"], "python": "3.12", "torch_version": "2.14.1", "torch_build": "cu130",
        "bundle_size": 5_000_000_000}
    part = pb.BlobPart(data=b"\x00\xff" * 10, offset=3, total_size=-1, last=True, digest="d", meta_json="{}")
    assert L.decode_message(part.SerializeToString(), L.PART_FIELDS) == {
        "data": b"\x00\xff" * 10, "offset": 3, "total_size": -1, "last": True, "digest": "d",
        "not_modified": False, "meta_json": "{}"}
    assert pb.BundleRequest.FromString(L.encode_bundle_request("h" * 64)).code_hash == "h" * 64
    for fields, message in ((L.INFO_FIELDS, pb.BootstrapInfo), (L.PART_FIELDS, pb.BlobPart)):
        assert {n: name for n, (name, _) in fields.items()} == {
            f.number: f.name for f in message.DESCRIPTOR.fields if f.name in {v[0] for v in fields.values()}}
    assert pb.BundleRequest.DESCRIPTOR.fields_by_name["code_hash"].number == L.BUNDLE_REQUEST_CODE_HASH
    # unknown fields of a newer hub are skipped
    extra = (msg.SerializeToString() + L._encode_varint(10 << 3) + b"\x07"
             + L._encode_varint(99 << 3 | 2) + b"\x01A")
    assert L.decode_message(extra, L.INFO_FIELDS)["python"] == "3.12"
    with pytest.raises(ValueError):
        L.decode_message(b"\x1a\x05ab", L.INFO_FIELDS)


def _rsa_cert():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "x" * 60)])     # RSA-2048: multi-byte DER lengths
    import datetime
    now = datetime.datetime.now(datetime.UTC)
    return (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=1)).sign(key, hashes.SHA256()))


def test_spki_fingerprint_matches_security(tmp_path):
    identity = HubIdentity.load_or_create(tmp_path / "hub-identity")
    der = x509.load_pem_x509_certificate(identity.ca_pem).public_bytes(serialization.Encoding.DER)
    assert L.spki_sha256(der) == identity.fingerprint == spki_fingerprint(identity.ca_pem)
    cert = _rsa_cert()
    spki = cert.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    assert L.spki_sha256(cert.public_bytes(serialization.Encoding.DER)) == hashlib.sha256(spki).hexdigest()
    assert L.curl_pin("00" * 32) == "sha256//" + "A" * 43 + "="


def test_token_and_address_parsing(tmp_path):
    token = str(HubIdentity.load_or_create(tmp_path / "a").token)
    f = tmp_path / "tok"
    f.write_text(token + "\n")
    env = {"COLOSSEUM_TOKEN": token}
    assert L.resolve_token(None, None, env) == (token, token.split(".")[2])
    assert L.resolve_token(None, str(f), {"COLOSSEUM_TOKEN": "bad"})[0] == token
    other = str(HubIdentity.load_or_create(tmp_path / "b").token)
    assert L.resolve_token(other, str(f), env)[0] == other
    for bad in ("", "cst1.short.abc", token + "x"):
        with pytest.raises(L.LoaderError) as e:
            L.resolve_token(bad or None, None, {})
        assert e.value.code == 2
    assert L.parse_address("10.0.0.5") == ("10.0.0.5", 47100)
    assert L.parse_address("hub.lan:5000") == ("hub.lan", 5000)
    assert L.parse_address("[::1]:47100") == ("::1", 47100)
    for bad in ("host:x", ":5", "h:70000"):
        with pytest.raises(L.LoaderError):
            L.parse_address(bad)


# --- end to end with exec_fn ------------------------------------------------------------------------------------

def test_join_builds_env_and_execs_the_agent(hub, tmp_path, monkeypatch):
    port = hub.start()
    env, _ = _env(tmp_path)
    code, calls = _run([f"127.0.0.1:{port}", "--workers", "3", "--tags", "gpu,big", "--once"],
                       {**env, "COLOSSEUM_TOKEN": hub.token}, monkeypatch)
    assert code == 0 and len(calls) == 1
    path, args, agent_env = calls[0]
    cache = tmp_path / "cache" / "colosseum"
    bundle_dir = cache / "bundles" / hub.service.bundle.code_hash
    env_dir = Path(path).parents[1]
    assert env_dir.parent == cache / "envs" and (env_dir / L.ENV_MARKER).is_file()
    assert args == [path, "-m", "colosseum.node.agent", f"127.0.0.1:{port}", "--workers", "3", "--tags", "gpu,big",
                    "--once", "--code-hash", hub.service.bundle.code_hash, "--bundle", str(bundle_dir)]
    assert hub.token not in " ".join(args)
    assert agent_env["COLOSSEUM_TOKEN"] == hub.token
    assert agent_env["PYTHONPATH"] == str(bundle_dir / "project")             # the caller's PYTHONPATH is gone
    assert agent_env["PYTHONNOUSERSITE"] == "1"
    assert Path.cwd() == bundle_dir / "project"
    venv_call, pip_call = _uv_calls(tmp_path)
    assert venv_call[:4] == ["venv", "--quiet", "--python", "3.12"]
    assert pip_call[:2] == ["pip", "install"] and pip_call[-2:] == ["--torch-backend", "cpu"]   # no GPU here
    req = (env_dir / "colosseum-requirements.txt").read_text().splitlines()
    assert req == [*join_kit.REQUIREMENTS, "torch==2.14.1"]
    uv_env = json.loads((tmp_path / "uv.jsonl").read_text().splitlines()[0])["env"]
    assert uv_env["UV_CACHE_DIR"] == str(cache / "uv-cache")
    assert uv_env["UV_PYTHON_INSTALL_DIR"] == str(cache / "python")


def test_second_join_reuses_bundle_and_env(hub, tmp_path, monkeypatch):
    port = hub.start()
    env, _ = _env(tmp_path, COLOSSEUM_TOKEN=hub.token)
    assert _run([f"127.0.0.1:{port}"], env, monkeypatch)[0] == 0
    first = _uv_calls(tmp_path)
    code, calls = _run([f"127.0.0.1:{port}"], env, monkeypatch)
    assert code == 0 and _uv_calls(tmp_path) == first                           # no uv call the second time
    gpu = T.NvidiaGpu(0, "RTX", 12282, "610.88", "8.9")
    code, calls = _run([f"127.0.0.1:{port}"], env, monkeypatch, gpus=[gpu])    # another build: another env
    assert code == 0 and len(_uv_calls(tmp_path)) == len(first) + 2
    pip = _uv_calls(tmp_path)[-1]
    assert pip[-2:] == ["--torch-backend", "cu130"]
    assert json.loads((tmp_path / "uv.jsonl").read_text().splitlines()[-1])["env"]["UV_CUDA_DRIVER_VERSION"] == "610.88"
    assert len(list((tmp_path / "cache" / "colosseum" / "envs").glob("*/" + L.ENV_MARKER))) == 2


def test_gpu_node_with_cpu_torch_warns(hub, tmp_path, monkeypatch, capsys):
    port = hub.start()
    env, _ = _env(tmp_path, COLOSSEUM_TOKEN=hub.token)
    gpu = T.NvidiaGpu(0, "RTX", 12282, "610.88", "8.9")
    assert _run([f"127.0.0.1:{port}"], env, monkeypatch, gpus=[gpu])[0] == 0
    assert "CPU build of torch was installed" in capsys.readouterr().err


def test_failed_install_names_the_package_and_removes_the_env(hub, tmp_path, monkeypatch, capsys):
    port = hub.start()
    env, _ = _env(tmp_path, COLOSSEUM_TOKEN=hub.token,
                  FAKE_UV_FAIL="error: Distribution `box2d==2.3.10` can't be installed because it doesn't have a "
                               "wheel for the current platform")
    code, calls = _run([f"127.0.0.1:{port}"], env, monkeypatch)
    err = capsys.readouterr().err
    assert code == 1 and not calls
    assert "box2d==2.3.10" in err and "distributed.requirements" in err
    assert not any((tmp_path / "cache" / "colosseum" / "envs").glob("*/bin"))


def test_wrong_torch_version_in_env_is_refused(hub, tmp_path, monkeypatch, capsys):
    port = hub.start()
    env, _ = _env(tmp_path, COLOSSEUM_TOKEN=hub.token, FAKE_TORCH_JSON=json.dumps(
        {"python": "3.12", "torch": "2.13.0+cpu", "cuda": None, "cuda_available": False, "mps": False}))
    code, _ = _run([f"127.0.0.1:{port}"], env, monkeypatch)
    assert code == 1 and "torch 2.13.0+cpu" in capsys.readouterr().err


def test_node_python_override_skips_the_env(hub, tmp_path, monkeypatch):
    port = hub.start()
    env, py = _env(tmp_path, COLOSSEUM_TOKEN=hub.token)
    env[L.NODE_PYTHON_ENV] = str(py)
    code, calls = _run([f"127.0.0.1:{port}"], env, monkeypatch)
    assert code == 0 and calls[0][0] == str(py) and _uv_calls(tmp_path) == []


@pytest.mark.parametrize("case, needle", [
    ("fingerprint", "does not have the key of this token"),
    ("secret", "refused the token"),
    ("insecure_hub", "--insecure"),
    ("tls_hub_insecure_node", "drop --insecure"),
    ("no_bootstrap", "colosseum train --listen"),
    ("down", "cannot reach the hub"),
])
def test_refusals(hub, tmp_path, monkeypatch, capsys, case, needle):
    env, _ = _env(tmp_path)
    token, argv = hub.token, []
    if case == "fingerprint":
        port = hub.start()
        token = str(HubIdentity.load_or_create(tmp_path / "other-identity").token)
    elif case == "secret":
        port = hub.start()
        token = "cst1." + "Z" * 43 + "." + hub.identity.fingerprint
    elif case == "insecure_hub":
        port = hub.start(tls=False)
    elif case == "tls_hub_insecure_node":
        port = hub.start()
        argv = ["--insecure"]
    elif case == "no_bootstrap":                                   # a plain `train` hub: no Bootstrap service
        port = hub.start(tls=False, service=None)
        argv = ["--insecure"]
    else:
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
    code, calls = _run([f"127.0.0.1:{port}", *argv], {**env, "COLOSSEUM_TOKEN": token}, monkeypatch)
    assert code == 1 and not calls
    assert needle in capsys.readouterr().err


def test_prune_keeps_recent_and_in_use(tmp_path):
    root = tmp_path / "envs"
    import os
    for i, name in enumerate(["a", "b", "c", "d"]):
        (root / name).mkdir(parents=True)
        (root / name / L.IN_USE).write_text("")
        os.utime(root / name / L.IN_USE, (100 + i, 100 + i))
    held = L._hold(root / "a" / L.IN_USE, exclusive=False)       # a running node holds the oldest env
    try:
        assert sorted(L.prune(root, 2, root / "d")) == ["b"]
    finally:
        held.close()
    assert sorted(p.name for p in root.iterdir()) == ["a", "c", "d"]


def test_standalone_copy_runs_without_colosseum(hub, tmp_path):
    """The installer's copy: loader.py with bundle.py and torch_select.py, under python -I."""
    port = hub.start()
    boot = tmp_path / "boot"
    boot.mkdir()
    for module in (L, B, T):
        shutil.copy(module.__file__, boot)
    env, py = _env(tmp_path, COLOSSEUM_TOKEN=hub.token)
    env[L.NODE_PYTHON_ENV] = str(py)
    out = subprocess.run([sys.executable, "-I", str(boot / "loader.py"), f"127.0.0.1:{port}", "--name", "lap"],
                         env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    record = json.loads((tmp_path / "record.jsonl").read_text())
    assert record["argv"][1:4] == ["-m", "colosseum.node.agent", f"127.0.0.1:{port}"]
    assert "--name" in record["argv"] and record["cwd"].endswith(f"{hub.service.bundle.code_hash}/project")


def test_bundle_in_several_parts(hub, tmp_path, monkeypatch):
    """A bundle larger than one blob part (send_parts) arrives whole and verified."""
    port = hub.start(extra_files={"data/noise.bin": os.urandom(3 * (1 << 20))})
    assert hub.service.bundle.size > 2 * (1 << 20)
    env, py = _env(tmp_path, COLOSSEUM_TOKEN=hub.token)
    env[L.NODE_PYTHON_ENV] = str(py)
    code, calls = _run([f"127.0.0.1:{port}"], env, monkeypatch)
    bundle_dir = tmp_path / "cache" / "colosseum" / "bundles" / hub.service.bundle.code_hash
    assert code == 0 and (bundle_dir / "project" / "data" / "noise.bin").stat().st_size == 3 * (1 << 20)


def test_info_of_the_real_service_decodes(tmp_path):
    service = join_kit.tiny_service(tmp_path)
    info = L.decode_message(service.info().SerializeToString(), L.INFO_FIELDS)
    assert info["code_hash"] == service.bundle.code_hash and info["requirements"] == list(join_kit.REQUIREMENTS)
    assert info["bundle_size"] == service.bundle.size and info["protocol_major"] == PROTOCOL_VERSION[0]
```

Create `tests/unit/test_sp5_bootstrap_service.py`:
```python
"""What a hub serves to joining machines (SP5 T6.2, spec 2.3-2.4): the BootstrapService built from a project."""
from __future__ import annotations

import os

import join_kit
from colosseum.bootstrap import installer as I
from colosseum.bootstrap import loader as L
from colosseum.net.messages import PROTOCOL_VERSION


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
```

Create `tests/integration/test_sp5_join_installer.py` (real `sh` / `dash` / `bash` and the real `curl`; the curl tests skip without a curl that has HTTP/2):
```python
"""The installer script and the curl one-liner (SP5 T6.2, spec 2.2; R2): end marker, truncation safety, embedded
files, the uv / loader chain with a fake uv, and the real curl against a real gRPC Bootstrap/Script."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import join_kit
from colosseum.bootstrap import bundle as B
from colosseum.bootstrap import installer as I
from colosseum.bootstrap import loader as L
from colosseum.bootstrap import torch_select as T
from colosseum.net.security import HubIdentity

SHELLS = [s for s in ("sh", "dash", "bash") if shutil.which(s)]


def _script(insecure=False) -> bytes:
    return I.installer_script(insecure=insecure, python="3.12", grpcio="1.84.0")


def test_script_shape_and_embedded_files(tmp_path):
    text = _script().decode()
    assert text.startswith("#!/bin/sh\n")
    assert text.endswith('_colosseum_install "$@" # colosseum-installer-end\n')
    assert text.count("_colosseum_install") == 2                         # the definition and the one call
    assert 'run --no-project --python 3.12 --with "grpcio==1.84.0" --' in text
    assert "https://astral.sh/uv/0.13.1/install.sh" in text and "UV_UNMANAGED_INSTALL" in text
    def tail(script: str) -> str:
        return script.rsplit('exec "$uv" run', 1)[1]
    assert "--insecure" not in tail(text) and "--insecure " in tail(_script(True).decode())
    assert _script() == _script()                                          # deterministic (served bytes)


def _env(tmp_path, port, token, **extra):
    uv, py = join_kit.write_fakes(tmp_path / "fakes")
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path / "home"), "XDG_CACHE_HOME": str(tmp_path / "cache"),
           "FAKE_RECORD": str(tmp_path / "record.jsonl"), "COLOSSEUM_UV": str(uv),
           "COLOSSEUM_HUB": f"127.0.0.1:{port}", "COLOSSEUM_TOKEN": token, L.NODE_PYTHON_ENV: str(py)}
    env.update(extra)
    return env


@pytest.fixture
def hub(tmp_path):
    identity = HubIdentity.load_or_create(tmp_path / "hub-identity")
    servers = []

    def start(tls=True):
        server, port = join_kit.bootstrap_server(join_kit.tiny_service(tmp_path, insecure=not tls), identity, tls=tls)
        servers.append(server)
        return port
    yield type("Hub", (), {"identity": identity, "token": str(identity.token), "start": staticmethod(start)})
    for s in servers:
        s.stop(0)


@pytest.mark.parametrize("shell", SHELLS)
def test_script_runs_the_loader_and_passes_join_flags(hub, tmp_path, shell):
    port = hub.start()
    env = _env(tmp_path, port, hub.token)
    out = subprocess.run([shell, "-s", "--", "--workers", "2", "--once"], input=_script(), env=env,
                         capture_output=True, timeout=60)
    assert out.returncode == 0, out.stderr.decode()
    record = json.loads((tmp_path / "record.jsonl").read_text())
    assert record["argv"][3:7] == [f"127.0.0.1:{port}", "--workers", "2", "--once"]
    boot = next((tmp_path / "cache" / "colosseum" / "loader").iterdir())
    for module in (L, B, T):
        assert (boot / Path(module.__file__).name).read_bytes() == Path(module.__file__).read_bytes()


LAST_LINE = b'_colosseum_install "$@" # colosseum-installer-end\n'


@pytest.mark.parametrize("cut", [0.05, 0.3, 0.5, 0.8, 0.99, "in-call", "before-call"])
def test_truncated_script_does_nothing(tmp_path, cut):
    """Every cut before the last line (the call) runs nothing: the function body is parsed whole first."""
    script = _script()
    if cut == "in-call":
        end = len(script) - len(LAST_LINE) + len(b"_colosseum_ins")
    elif cut == "before-call":
        end = len(script) - len(LAST_LINE)
    else:
        end = int(len(script) * cut)
    env = _env(tmp_path, 1, "x")
    for shell in SHELLS:
        subprocess.run([shell], input=script[:end], env=env, capture_output=True, timeout=30)
    assert not (tmp_path / "cache").exists() and not (tmp_path / "record.jsonl").exists()


def test_missing_hub_or_token_is_explained(tmp_path):
    env = _env(tmp_path, 1, "x")
    del env["COLOSSEUM_HUB"]
    out = subprocess.run(["sh"], input=_script(), env=env, capture_output=True, timeout=30)
    assert out.returncode == 2 and b"COLOSSEUM_HUB is not set" in out.stderr


def _curl_http2() -> bool:
    exe = shutil.which("curl")
    return bool(exe) and "HTTP2" in subprocess.run([exe, "--version"], capture_output=True, text=True).stdout


@pytest.mark.skipif(not _curl_http2(), reason="needs curl with HTTP/2")
@pytest.mark.parametrize("tls", [True, False])
def test_one_liner_end_to_end_with_real_curl(hub, tmp_path, tls):
    port = hub.start(tls=tls)
    address = f"127.0.0.1:{port}"
    line = I.one_liner(address, hub.token, insecure=not tls)
    env = _env(tmp_path, port, "unused")
    del env["COLOSSEUM_HUB"], env["COLOSSEUM_TOKEN"]                      # the one-liner sets both
    out = subprocess.run(["sh", "-c", line], env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    record = json.loads((tmp_path / "record.jsonl").read_text())
    assert record["argv"][3] == address and record["env"]["COLOSSEUM_TOKEN"] == hub.token
    assert ("--insecure" in record["argv"]) == (not tls)


@pytest.mark.skipif(not _curl_http2(), reason="needs curl with HTTP/2")
def test_one_liner_with_a_wrong_pin_runs_nothing(hub, tmp_path):
    port = hub.start()
    other = HubIdentity.load_or_create(tmp_path / "other-identity")
    token = f"cst1.{hub.identity.token.secret}.{other.fingerprint}"      # pins a key the hub does not have
    line = I.one_liner(f"127.0.0.1:{port}", token, insecure=False)
    out = subprocess.run(["sh", "-c", line], env=_env(tmp_path, port, token), capture_output=True, text=True,
                         timeout=60)
    assert "(90)" in out.stderr                                          # curl: public key does not match
    assert not (tmp_path / "record.jsonl").exists() and not (tmp_path / "cache").exists()


def test_one_liner_and_join_command_text():
    token = "cst1." + "A" * 43 + "." + "0" * 64
    line = I.one_liner("10.0.0.5:47100", token, insecure=False)
    assert line == ("printf '\\0\\0\\0\\0\\0' | curl -fsS --http2 -k --pinnedpubkey 'sha256//" + "A" * 43 + "=' "
                    "-H 'content-type: application/grpc' -H 'te: trailers' --data-binary @- "
                    "https://10.0.0.5:47100/colosseum.v1.Bootstrap/Script | tail -c +6 | "
                    f"COLOSSEUM_HUB='10.0.0.5:47100' COLOSSEUM_TOKEN='{token}' sh")
    assert I.join_command("10.0.0.5:47100", token, insecure=True) == \
        f"colosseum join 10.0.0.5:47100 --token '{token}' --insecure"
```

- [ ] **Step 3: Run them (RED)**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_join_loader.py tests/unit/test_sp5_bootstrap_service.py tests/integration/test_sp5_join_installer.py -q`
Expected: collection errors (`ModuleNotFoundError: No module named 'colosseum.bootstrap.loader'`, `ImportError: cannot import name 'add_bootstrap' from 'colosseum.hub.server'`).

- [ ] **Step 4: Write `loader.py`**

`src/colosseum/bootstrap/loader.py`:
```python
"""The thin ``join`` loader (SP5 spec 2.3): from a hub address and a token to a running node agent.

Steps: check the hub's certificate against the token's fingerprint (TLS; skipped with ``--insecure``) ->
``Bootstrap/Info`` -> the code bundle by hash (``Bootstrap/Bundle``, token in the metadata; cached in
``<cache>/colosseum/bundles/<hash>/``) -> the node environment (uv: the hub's Python major.minor, the hub's
requirement list, ``torch==<hub version>`` with the build of this machine; cached in
``<cache>/colosseum/envs/<key>/``) -> ``exec`` of ``python -m colosseum.node.agent`` from that environment, with
``PYTHONPATH`` on the bundle and the bundle's project directory as the working directory.

The file is self-contained (standard library + ``grpcio``; no protobuf, no colosseum imports): the installer script
embeds it verbatim next to ``bundle.py`` and ``torch_select.py`` (loaded from this file's directory), and
``colosseum join`` runs it from an installed colosseum of any version. It speaks only the ``Bootstrap`` service,
whose three messages it encodes and decodes itself (field numbers checked against the generated classes by
``tests/unit/test_sp5_loader.py``).
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

BOOTSTRAP = "/colosseum.v1.Bootstrap/"
MD_TOKEN = "x-colosseum-token"
HUB_SERVER_NAME = "colosseum-hub"
DEFAULT_PORT = 47100
TOKEN_ENV = "COLOSSEUM_TOKEN"
NODE_PYTHON_ENV = "COLOSSEUM_NODE_PYTHON"     # use this interpreter instead of building an environment
UV_ENV = "COLOSSEUM_UV"                       # the uv binary to use
TOKEN_RE = re.compile(r"^cst1\.([A-Za-z0-9_-]{43})\.([0-9a-f]{64})$")
UV_VERSION = "0.13.1"
MIN_UV = (0, 7, 14)                           # --torch-backend without --preview
UV_INSTALL_URL = "https://astral.sh/uv/{version}/install.sh"
CLEAN_ENV = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "VIRTUAL_ENV", "PYTHONSAFEPATH",
             "PYTHONNOUSERSITE")
ENV_MARKER = ".colosseum-env.json"
IN_USE = ".in-use"
KEEP_ENVS = 2
KEEP_BUNDLES = 5
CONNECT_TIMEOUT_SEC = 10.0
AGENT_MODULE = "colosseum.node.agent"
CHECK_SCRIPT = (
    "import json, sys\n"
    "try:\n"
    "    import torch\n"
    "except Exception as e:\n"
    "    print(json.dumps({'error': repr(e)})); raise SystemExit(0)\n"
    "mps = getattr(torch.backends, 'mps', None)\n"
    "print(json.dumps({'python': '%d.%d' % sys.version_info[:2], 'torch': torch.__version__,\n"
    "                  'cuda': torch.version.cuda, 'cuda_available': bool(torch.cuda.is_available()),\n"
    "                  'mps': bool(mps is not None and mps.is_available())}))\n"
)


class LoaderError(Exception):
    """A refusal or failure of the loader; ``colosseum join`` prints it and exits with ``code``."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


def _say(message: str) -> None:
    print(f"colosseum join: {message}", file=sys.stderr, flush=True)


def _sibling(name: str):
    """``bundle`` / ``torch_select``: from the package when this file is ``colosseum.bootstrap.loader``, else the
    file next to this one (installer copy)."""
    if __package__:
        return importlib.import_module(f"{__package__}.{name}")
    module_name = f"_colosseum_loader_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, Path(__file__).with_name(f"{name}.py"))
    if spec is None or spec.loader is None:
        raise LoaderError(f"{name}.py is missing next to {__file__}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------------------------
# minimal protobuf wire codec for BootstrapInfo, BundleRequest, BlobPart
# ---------------------------------------------------------------------------------------------------------------

INFO_FIELDS = {1: ("protocol_major", "int"), 2: ("protocol_minor", "int"), 3: ("code_hash", "str"),
               4: ("requirements_hash", "str"), 5: ("requirements", "strs"), 6: ("python", "str"),
               7: ("torch_version", "str"), 8: ("torch_build", "str"), 9: ("bundle_size", "int")}
PART_FIELDS = {1: ("data", "bytes"), 2: ("offset", "int"), 3: ("total_size", "int"), 4: ("last", "bool"),
               5: ("digest", "str"), 6: ("not_modified", "bool"), 7: ("meta_json", "str")}
BUNDLE_REQUEST_CODE_HASH = 1


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(buf) or shift > 63:
            raise ValueError("truncated varint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _encode_varint(n: int) -> bytes:
    n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def decode_message(buf: bytes, fields: Mapping[int, tuple[str, str]]) -> dict:
    """Decode the known fields of a protobuf message (unknown fields are skipped: newer hubs may add some)."""
    out: dict = {name: ([] if kind == "strs" else b"" if kind == "bytes" else "" if kind == "str"
                        else False if kind == "bool" else 0) for name, kind in fields.values()}
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 2:
            length, pos = _varint(buf, pos)
            if pos + length > len(buf):
                raise ValueError("truncated field")
            value, pos = buf[pos:pos + length], pos + length
        elif wire == 1:
            value, pos = buf[pos:pos + 8], pos + 8
        elif wire == 5:
            value, pos = buf[pos:pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        if number not in fields:
            continue
        name, kind = fields[number]
        if kind == "int":
            out[name] = value - (1 << 64) if value >= 1 << 63 else value
        elif kind == "bool":
            out[name] = bool(value)
        elif kind == "str":
            out[name] = value.decode("utf-8")
        elif kind == "strs":
            out[name].append(value.decode("utf-8"))
        else:
            out[name] = bytes(value)
    return out


def encode_bundle_request(code_hash: str) -> bytes:
    data = code_hash.encode()
    return _encode_varint(BUNDLE_REQUEST_CODE_HASH << 3 | 2) + _encode_varint(len(data)) + data


# ---------------------------------------------------------------------------------------------------------------
# token, address, certificate
# ---------------------------------------------------------------------------------------------------------------

def resolve_token(token: str | None, token_file: str | None, environ: Mapping[str, str]) -> tuple[str, str]:
    """(token, fingerprint): ``--token`` > ``--token-file`` > ``COLOSSEUM_TOKEN``."""
    text = token
    if text is None and token_file is not None:
        try:
            text = Path(token_file).read_text().strip()
        except OSError as e:
            raise LoaderError(f"cannot read the token file {token_file}: {e}", 2) from e
    if text is None:
        text = environ.get(TOKEN_ENV)
    if not text:
        raise LoaderError("no token: pass --token, --token-file or set COLOSSEUM_TOKEN "
                          "(colosseum hub token prints it on the hub machine)", 2)
    match = TOKEN_RE.match(text.strip())
    if match is None:
        raise LoaderError("the token is malformed (expected cst1.<secret>.<fingerprint>; copy it again from "
                          "colosseum hub token)", 2)
    return text.strip(), match.group(2)


def parse_address(address: str) -> tuple[str, int]:
    """``host``, ``host:port``, ``[v6]:port`` -> (host, port); the port defaults to 47100."""
    text = address.strip()
    if text.startswith("["):
        host, _, rest = text[1:].partition("]")
        port = rest[1:] if rest.startswith(":") else ""
    elif text.count(":") == 1:
        host, port = text.split(":")
    else:
        host, port = text, ""
    if not host or (port and not port.isdigit()) or (port and not 0 < int(port) < 65536):
        raise LoaderError(f"bad hub address {address!r} (expected host:port)", 2)
    return host, int(port) if port else DEFAULT_PORT


def _tlv(buf: bytes, pos: int) -> tuple[int, int, int]:
    """(tag, content start, content end) of the DER element at ``pos``."""
    tag = buf[pos]
    length = buf[pos + 1]
    pos += 2
    if length & 0x80:
        count = length & 0x7F
        if not 0 < count <= 4:
            raise ValueError("bad DER length")
        length = int.from_bytes(buf[pos:pos + count], "big")
        pos += count
    if pos + length > len(buf):
        raise ValueError("truncated DER")
    return tag, pos, pos + length


def spki_sha256(cert_der: bytes) -> str:
    """SHA-256 hex of the certificate's SubjectPublicKeyInfo (DER), as in the token."""
    tag, start, _ = _tlv(cert_der, 0)                  # Certificate
    tag, pos, end = _tlv(cert_der, start)              # tbsCertificate
    if tag != 0x30:
        raise ValueError("not a certificate")
    fields = []
    while pos < end:
        t, s, e = _tlv(cert_der, pos)
        fields.append((t, pos, e))
        pos = e
    if fields and fields[0][0] == 0xA0:                # [0] version
        fields = fields[1:]
    # serialNumber, signature, issuer, validity, subject, subjectPublicKeyInfo
    _, s, e = fields[5]
    return hashlib.sha256(cert_der[s:e]).hexdigest()


def curl_pin(fingerprint: str) -> str:
    return "sha256//" + base64.b64encode(bytes.fromhex(fingerprint)).decode()


def fetch_hub_ca(host: str, port: int, fingerprint: str, *, timeout: float) -> bytes:
    """The hub's certificate (R15: the CA itself) from an unverified handshake, accepted only when its key
    matches the token's fingerprint."""
    try:
        pem = ssl.get_server_certificate((host, port), timeout=timeout)
    except ssl.SSLError as e:
        raise LoaderError(f"TLS handshake with {host}:{port} failed ({e.reason or e}); if the hub runs with "
                          "--insecure, add --insecure here") from e
    except (TimeoutError, OSError) as e:
        raise LoaderError(f"cannot reach the hub at {host}:{port} ({e}); is it running, is the port open "
                          "(WSL2 hubs: docs/DISTRIBUTED_GUIDE.md, WSL2)?") from e
    actual = spki_sha256(ssl.PEM_cert_to_DER_cert(pem))
    if actual != fingerprint:
        raise LoaderError(f"the hub at {host}:{port} does not have the key of this token (fingerprint "
                          f"{actual[:16]}..., the token names {fingerprint[:16]}...): another hub, a rotated "
                          "token (colosseum hub token) or an interception")
    return pem.encode()


def open_channel(host: str, port: int, *, fingerprint: str, insecure: bool, max_message_mb: int = 64):
    import grpc

    target = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    options = [("grpc.max_receive_message_length", max_message_mb << 20),
               ("grpc.max_send_message_length", max_message_mb << 20)]
    if insecure:
        return grpc.insecure_channel(target, options=options)
    ca = fetch_hub_ca(host, port, fingerprint, timeout=CONNECT_TIMEOUT_SEC)
    options.append(("grpc.ssl_target_name_override", HUB_SERVER_NAME))
    return grpc.secure_channel(target, grpc.ssl_channel_credentials(root_certificates=ca), options=options)


def _rpc_error(e, what: str, *, insecure: bool = False) -> LoaderError:
    import grpc

    code = e.code() if hasattr(e, "code") else None
    if code == grpc.StatusCode.UNAUTHENTICATED:
        return LoaderError(f"{what}: the hub refused the token ({e.details()})")
    if code == grpc.StatusCode.PERMISSION_DENIED:
        return LoaderError(f"{what}: refused by the hub ({e.details()})")
    if code == grpc.StatusCode.UNIMPLEMENTED:
        return LoaderError(f"{what}: the hub serves no code to nodes (a plain `colosseum train` listens on "
                           "loopback only: start `colosseum hub` or `colosseum train --listen`)")
    if code == grpc.StatusCode.UNAVAILABLE:
        hint = ("unless the hub runs with --insecure, drop --insecure here" if insecure
                else "is the hub running and its port open?")
        return LoaderError(f"{what}: the hub is not reachable ({e.details()}); {hint}")
    return LoaderError(f"{what} failed: {code} {e.details() if hasattr(e, 'details') else e}")


def fetch_info(channel, *, insecure: bool = False) -> dict:
    import grpc

    call = channel.unary_unary(BOOTSTRAP + "Info", request_serializer=None, response_deserializer=None)
    try:
        raw = call(b"", timeout=CONNECT_TIMEOUT_SEC * 3)
    except grpc.RpcError as e:
        raise _rpc_error(e, "Bootstrap/Info", insecure=insecure) from e
    try:
        return decode_message(raw, INFO_FIELDS)
    except ValueError as e:
        raise LoaderError(f"the hub's Bootstrap/Info cannot be read by this loader ({e}); run the hub's "
                          "one-liner instead") from e


# ---------------------------------------------------------------------------------------------------------------
# cache: bundle, uv, environment
# ---------------------------------------------------------------------------------------------------------------

def cache_home(environ: Mapping[str, str]) -> Path:
    return Path(environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")


def _hold(path: Path, *, exclusive: bool, blocking: bool = True):
    """An flock on ``path`` (created); returns the open file object (keep it to keep the lock)."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+")
    try:
        fcntl.flock(f, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | (0 if blocking else fcntl.LOCK_NB))
    except OSError:
        f.close()
        raise
    return f


def prune(directory: Path, keep: int, current: Path) -> list[str]:
    """Remove all but the ``keep`` most recently used entries of ``directory`` (by their ``.in-use`` file's
    mtime); an entry another process holds (shared lock on ``.in-use``) or ``current`` is never removed."""
    removed = []
    entries = [p for p in directory.iterdir() if p.is_dir() and not p.name.startswith(".")] \
        if directory.is_dir() else []
    entries.sort(key=lambda p: (p / IN_USE).stat().st_mtime if (p / IN_USE).exists() else 0.0, reverse=True)
    for entry in entries[keep:]:
        if entry == current:
            continue
        try:
            lock = _hold(entry / IN_USE, exclusive=True, blocking=False)
        except OSError:
            continue                                   # in use by a running node
        try:
            shutil.rmtree(entry, ignore_errors=True)
            removed.append(entry.name)
        finally:
            lock.close()
    return removed


def _touch_in_use(entry: Path):
    lock = _hold(entry / IN_USE, exclusive=False)
    os.utime(entry / IN_USE)
    return lock


def ensure_bundle(channel, info: dict, token: str, cache: Path) -> Path:
    import grpc

    bundle = _sibling("bundle")
    code_hash = info["code_hash"]
    if not re.fullmatch(r"[0-9a-f]{64}", code_hash or ""):
        raise LoaderError(f"the hub announced a malformed code hash {code_hash!r}")
    target = bundle.bundles_dir(cache) / code_hash
    if (target / bundle.COMPLETE_MARKER).is_file():
        return target
    size = int(info["bundle_size"])
    if not 0 < size <= bundle.MAX_ARCHIVE_BYTES:
        raise LoaderError(f"the hub announced a code bundle of {size} bytes")
    _say(f"downloading the code bundle {code_hash[:12]} ({size / 1e6:.1f} MB)")
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.parent / f".dl-{code_hash[:16]}-{os.getpid()}.tar.gz"
    call = channel.unary_stream(BOOTSTRAP + "Bundle", request_serializer=None, response_deserializer=None)
    received = 0
    try:
        with open(part, "wb") as out:
            for raw in call(encode_bundle_request(code_hash), metadata=((MD_TOKEN, token.split(".")[1]),),
                            timeout=3600):
                piece = decode_message(raw, PART_FIELDS)
                if piece["offset"] != received:
                    raise LoaderError(f"code bundle: part at offset {piece['offset']}, expected {received}")
                received += len(piece["data"])
                if received > size:
                    raise LoaderError("code bundle: more data than announced")
                out.write(piece["data"])
                if piece["last"]:
                    break
        if received != size:
            raise LoaderError(f"code bundle: {received} of {size} bytes received")
        try:
            return bundle.unpack_bundle(part, code_hash, cache_root=cache)
        except bundle.BundleError as e:
            raise LoaderError(str(e)) from e
    except grpc.RpcError as e:
        raise _rpc_error(e, "Bootstrap/Bundle") from e
    finally:
        part.unlink(missing_ok=True)


def _uv_version(uv: str) -> tuple[int, ...] | None:
    try:
        out = subprocess.run([uv, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", out.stdout)
    return tuple(int(x) for x in match.groups()) if out.returncode == 0 and match else None


def ensure_uv(explicit: str | None, environ: Mapping[str, str], cache: Path) -> str:
    """``COLOSSEUM_UV`` > ``--uv`` > ``<cache>/colosseum/uv/uv`` > ``uv`` on PATH (>= 0.7.14) > a private uv
    installed with the pinned astral.sh installer (``UV_UNMANAGED_INSTALL``: no profile edits, no receipt)."""
    private = cache / "colosseum" / "uv" / "uv"
    for candidate in (environ.get(UV_ENV), explicit, str(private), shutil.which("uv", path=environ.get("PATH"))):
        if candidate and Path(candidate).exists() and (_uv_version(candidate) or (0,)) >= MIN_UV:
            return candidate
    _say(f"installing uv {UV_VERSION} into {private.parent}")
    url = UV_INSTALL_URL.format(version=UV_VERSION)
    fetch = f"curl -LsSf {url}" if shutil.which("curl") else f"wget -qO- {url}" if shutil.which("wget") else None
    if fetch is None:
        raise LoaderError("uv is not installed and neither curl nor wget is available to install it")
    env = {**environ, "UV_UNMANAGED_INSTALL": str(private.parent), "UV_NO_MODIFY_PATH": "1"}
    if subprocess.run(["sh", "-c", f"{fetch} | sh"], env=env).returncode != 0 or not private.exists():
        raise LoaderError(f"installing uv from {url} failed (install uv yourself and retry)")
    return str(private)


def env_key(info: dict, backend: str | None) -> str:
    text = "|".join([info["requirements_hash"], info["python"], info["torch_version"], backend or "default"])
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def _uv_environ(environ: Mapping[str, str], cache: Path, driver: str | None) -> dict[str, str]:
    torch_select = _sibling("torch_select")
    env = {k: v for k, v in torch_select.nvidia_smi_environ(environ).items() if k not in CLEAN_ENV}
    env.setdefault("UV_CACHE_DIR", str(cache / "colosseum" / "uv-cache"))
    env.setdefault("UV_PYTHON_INSTALL_DIR", str(cache / "colosseum" / "python"))
    if driver:
        env["UV_CUDA_DRIVER_VERSION"] = driver
    return env


def check_env(python: str, env: Mapping[str, str]) -> dict:
    out = subprocess.run([python, "-I", "-c", CHECK_SCRIPT], capture_output=True, text=True, env=dict(env),
                         timeout=300)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": (out.stderr or out.stdout).strip()[-500:]}


def ensure_env(info: dict, *, uv: str, cache: Path, environ: Mapping[str, str], choice, gpus) -> Path:
    """The node environment for ``info`` with ``choice`` (a ``torch_select.TorchChoice``): built once per
    (requirements, Python, torch version, backend), under an exclusive lock; the marker file is written last."""
    root = cache / "colosseum" / "envs"
    env_dir = root / env_key(info, choice.backend)
    python = env_dir / "bin" / "python"
    if (env_dir / ENV_MARKER).is_file() and python.exists():
        return env_dir
    root.mkdir(parents=True, exist_ok=True)
    with _hold(root / f"{env_dir.name}.lock", exclusive=True):
        if (env_dir / ENV_MARKER).is_file() and python.exists():
            return env_dir
        shutil.rmtree(env_dir, ignore_errors=True)
        env = _uv_environ(environ, cache, choice.cuda_driver)
        build = choice.backend or "the default build"
        _say(f"building the node environment (Python {info['python']}, {choice.requirement}, {build}); "
             "the first time this downloads torch")
        try:
            if subprocess.run([uv, "venv", "--quiet", "--python", info["python"], str(env_dir)], env=env).returncode:
                raise LoaderError(f"uv could not create a Python {info['python']} environment in {env_dir}")
            req = env_dir / "colosseum-requirements.txt"
            req.write_text("\n".join([*info["requirements"], choice.requirement]) + "\n")
            cmd = [uv, "pip", "install", "--python", str(python), "-r", str(req)]
            if choice.backend:
                cmd += ["--torch-backend", choice.backend]
            proc = subprocess.run(cmd, env=env, stderr=subprocess.PIPE, text=True)
            if proc.returncode:
                tail = "\n".join(proc.stderr.strip().splitlines()[-12:])
                sys.stderr.write(proc.stderr)
                raise LoaderError("installing the hub's requirements failed:\n" + tail + "\nA package without a "
                                  "wheel for this machine can be replaced on the hub with distributed.requirements "
                                  "(a requirements file of your own).")
            if proc.stderr:
                sys.stderr.write(proc.stderr)
            found = check_env(str(python), env)
            if "error" in found:
                raise LoaderError(f"the new environment cannot import torch: {found['error']}")
            if found["torch"].split("+", 1)[0] != info["torch_version"]:
                raise LoaderError(f"uv installed torch {found['torch']}, the hub runs {info['torch_version']}")
            if gpus and not found["cuda"]:
                _say(f"WARNING: this machine has an NVIDIA GPU but the CPU build of torch was installed (driver "
                     f"{choice.cuda_driver}); learners here run on the CPU. Update the driver to use the GPU.")
            elif gpus and not found["cuda_available"]:
                _say("WARNING: the CUDA build of torch was installed but CUDA is not available here")
            (env_dir / ENV_MARKER).write_text(json.dumps({**found, "backend": choice.backend,
                                                          "requirements_hash": info["requirements_hash"]}))
        except BaseException:
            shutil.rmtree(env_dir, ignore_errors=True)
            raise
    return env_dir


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="colosseum join", description="Join this machine to a Colosseum hub.")
    p.add_argument("address", help="the hub's host:port")
    p.add_argument("--token")
    p.add_argument("--token-file")
    p.add_argument("--workers", type=int)
    p.add_argument("--evaluators", type=int)
    p.add_argument("--cpu-learners", type=int)
    p.add_argument("--no-learners", action="store_true")
    p.add_argument("--gpus")
    p.add_argument("--name")
    p.add_argument("--tags")
    p.add_argument("--once", action="store_true")
    p.add_argument("--insecure", action="store_true")
    p.add_argument("--uv", help=argparse.SUPPRESS)
    return p


def agent_args(ns: argparse.Namespace) -> list[str]:
    """The node agent's arguments: the join arguments without the token and the loader-only flags."""
    out = [ns.address]
    for flag in ("workers", "evaluators", "cpu_learners", "gpus", "name", "tags"):
        value = getattr(ns, flag)
        if value is not None:
            out += [f"--{flag.replace('_', '-')}", str(value)]
    for flag in ("no_learners", "once", "insecure"):
        if getattr(ns, flag):
            out.append(f"--{flag.replace('_', '-')}")
    return out


def agent_environ(environ: Mapping[str, str], bundle_dir: Path, token: str) -> dict[str, str]:
    bundle = _sibling("bundle")
    torch_select = _sibling("torch_select")
    env = {k: v for k, v in torch_select.nvidia_smi_environ(environ).items() if k not in CLEAN_ENV}
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in bundle.bundle_python_path(bundle_dir))
    env["PYTHONNOUSERSITE"] = "1"
    env[TOKEN_ENV] = token
    return env


def loader_main(argv: Sequence[str], *, environ: Mapping[str, str] | None = None,
                exec_fn: Callable[[str, list[str], dict[str, str]], object] | None = None) -> int:
    environ = dict(os.environ if environ is None else environ)
    exec_fn = exec_fn or os.execve
    try:
        ns = build_parser().parse_args(list(argv))
    except SystemExit as e:
        return int(e.code or 0)
    locks: list = []                                   # .in-use locks: inherited by the agent through exec
    try:
        if os.name != "posix":
            raise LoaderError("native Windows is not supported: run colosseum join inside WSL2")
        token, fingerprint = resolve_token(ns.token, ns.token_file, environ)
        host, port = parse_address(ns.address)
        ns.address = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
        bundle = _sibling("bundle")
        torch_select = _sibling("torch_select")
        cache = cache_home(environ)
        channel = open_channel(host, port, fingerprint=fingerprint, insecure=ns.insecure)
        with contextlib.closing(channel):
            info = fetch_info(channel, insecure=ns.insecure)
            bundle_dir = ensure_bundle(channel, info, token, cache)
        locks.append(_touch_in_use(bundle_dir))
        prune(bundle.bundles_dir(cache), KEEP_BUNDLES, bundle_dir)
        python = environ.get(NODE_PYTHON_ENV)
        if not python:
            gpus = torch_select.probe_nvidia(environ=environ)
            try:
                choice = torch_select.torch_spec(info["torch_version"], info["torch_build"],
                                                 platform=torch_select.probe_platform(), gpus=gpus)
            except torch_select.TorchSelectError as e:
                raise LoaderError(str(e)) from e
            if choice.warning:
                _say(f"WARNING: {choice.warning}")
            uv = ensure_uv(ns.uv, environ, cache)
            env_dir = ensure_env(info, uv=uv, cache=cache, environ=environ, choice=choice, gpus=gpus)
            locks.append(_touch_in_use(env_dir))
            prune(env_dir.parent, KEEP_ENVS, env_dir)
            python = str(env_dir / "bin" / "python")
        for lock in locks:
            os.set_inheritable(lock.fileno(), True)    # the agent keeps both in use for its lifetime
        args = [python, "-m", AGENT_MODULE, *agent_args(ns), "--code-hash", info["code_hash"],
                "--bundle", str(bundle_dir)]
        os.chdir(bundle.project_dir(bundle_dir))
        _say(f"starting the node agent (code {info['code_hash'][:12]})")
        exec_fn(python, args, agent_environ(environ, bundle_dir, token))
        return 0
    except LoaderError as e:
        _say(str(e))
        return e.code
    except KeyboardInterrupt:
        _say("interrupted")
        return 130
    finally:                                           # reached only when exec did not replace the process
        for lock in locks:
            lock.close()


if __name__ == "__main__":
    sys.exit(loader_main(sys.argv[1:]))
```

- [ ] **Step 5: Write `installer.py`**

`src/colosseum/bootstrap/installer.py`:
```python
"""The installer script and the join commands a hub prints (SP5 spec 2.2).

``installer_script`` is what ``Bootstrap/Script`` serves (raw bytes, R2): a POSIX ``sh`` script whose whole body is
the function ``_colosseum_install``, called by its last line (``... # colosseum-installer-end``), so a script cut
anywhere does nothing. It writes the embedded ``loader.py``, ``bundle.py`` and ``torch_select.py`` into
``<cache>/colosseum/loader/<id>/``, finds or installs uv (``COLOSSEUM_UV``, the private
``<cache>/colosseum/uv/uv``, ``uv`` on PATH, else the pinned astral.sh installer with ``UV_UNMANAGED_INSTALL``) and
runs the loader with the hub's Python and grpcio through ``uv run``. The hub address and the token come from the
environment (``COLOSSEUM_HUB``, ``COLOSSEUM_TOKEN``) set by the one-liner, extra arguments are ``join`` flags
(``sh -s -- --workers 4``), so one script serves every address of the hub and the SSH mode (T8.1).
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from colosseum.bootstrap import bundle as _bundle
from colosseum.bootstrap import loader as _loader
from colosseum.bootstrap import torch_select as _torch_select

END_MARKER = "# colosseum-installer-end"
SCRIPT_PATH = "/colosseum.v1.Bootstrap/Script"
EMBEDDED = (_loader, _bundle, _torch_select)
KEEP_HUB_BUNDLES = 5

_TEMPLATE = r"""#!/bin/sh
# Colosseum node installer, served by the hub (Bootstrap/Script). Run the command the hub printed:
#   ... | COLOSSEUM_HUB=host:port COLOSSEUM_TOKEN=cst1... sh [-s -- join flags]
_colosseum_install() {
    set -eu
    hub="${COLOSSEUM_HUB:-}"
    if [ -z "$hub" ]; then
        echo "colosseum: COLOSSEUM_HUB is not set (copy the whole command the hub printed)" >&2; return 2
    fi
    if [ -z "${COLOSSEUM_TOKEN:-}" ]; then
        echo "colosseum: COLOSSEUM_TOKEN is not set (copy the whole command the hub printed)" >&2; return 2
    fi
    case "$(uname -s)" in
        Linux|Darwin) ;;
        *) echo "colosseum: $(uname -s) is not supported (Linux, macOS, or Windows via WSL2)" >&2; return 1 ;;
    esac
    cache="${XDG_CACHE_HOME:-$HOME/.cache}/colosseum"
    boot="$cache/loader/@ID@"
    mkdir -p "$boot"
@FILES@
    uv="${COLOSSEUM_UV:-}"
    if [ -z "$uv" ] && [ -x "$cache/uv/uv" ]; then uv="$cache/uv/uv"; fi
    if [ -z "$uv" ] && command -v uv >/dev/null 2>&1; then uv="$(command -v uv)"; fi
    if [ -z "$uv" ]; then
        echo "colosseum: installing uv @UV@ into $cache/uv" >&2
        if command -v curl >/dev/null 2>&1; then
            curl -LsSf "https://astral.sh/uv/@UV@/install.sh" \
                | env UV_UNMANAGED_INSTALL="$cache/uv" UV_NO_MODIFY_PATH=1 sh
        elif command -v wget >/dev/null 2>&1; then
            wget -qO- "https://astral.sh/uv/@UV@/install.sh" \
                | env UV_UNMANAGED_INSTALL="$cache/uv" UV_NO_MODIFY_PATH=1 sh
        else
            echo "colosseum: neither curl nor wget is available to install uv" >&2; return 1
        fi
        uv="$cache/uv/uv"
    fi
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$cache/uv-cache}"
    export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$cache/python}"
    unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV
    exec "$uv" run --no-project --python @PYTHON@ --with "grpcio==@GRPCIO@" -- \
        python -I "$boot/loader.py" --uv "$uv" @INSECURE@"$hub" "$@"
}
_colosseum_install "$@" @END@
"""


def _embedded_files() -> list[tuple[str, str]]:
    return [(Path(m.__file__).name, Path(m.__file__).read_text()) for m in EMBEDDED]


def installer_script(*, insecure: bool, python: str, grpcio: str, uv_version: str = _loader.UV_VERSION) -> bytes:
    """The installer script for a hub with Python ``python`` (major.minor) and grpcio ``grpcio``."""
    files = _embedded_files()
    ident = hashlib.sha256("\0".join(f"{n}\0{t}" for n, t in files).encode()).hexdigest()[:16]
    blocks = []
    for name, text in files:
        tag = "COLOSSEUM_EOF_" + hashlib.sha256(text.encode()).hexdigest()[:12].upper()
        assert tag not in text and text.endswith("\n")
        blocks.append(f"    cat > \"$boot/{name}\" <<'{tag}'\n{text}{tag}\n")
    script = (_TEMPLATE.replace("@FILES@", "".join(blocks).rstrip("\n"))
              .replace("@ID@", ident).replace("@UV@", uv_version).replace("@PYTHON@", python)
              .replace("@GRPCIO@", grpcio).replace("@INSECURE@", "--insecure " if insecure else "")
              .replace("@END@", END_MARKER))
    return script.encode()


def _sq(text: str) -> str:
    """Single-quoted for sh; the values (address, pin, token) never contain a quote (checked)."""
    if "'" in text:
        raise ValueError(f"cannot quote {text!r}")
    return f"'{text}'"


def curl_pin(fingerprint: str) -> str:
    return "sha256//" + base64.b64encode(bytes.fromhex(fingerprint)).decode()


def one_liner(address: str, token: str, *, insecure: bool) -> str:
    """The spec 2.2 command for a machine without colosseum (``address`` = host:port as the node reaches it)."""
    fingerprint = str(token).split(".")[2]
    if insecure:
        transport, url = "--http2-prior-knowledge", f"http://{address}{SCRIPT_PATH}"
    else:
        transport, url = f"--http2 -k --pinnedpubkey {_sq(curl_pin(fingerprint))}", f"https://{address}{SCRIPT_PATH}"
    return (f"printf '\\0\\0\\0\\0\\0' | curl -fsS {transport} -H 'content-type: application/grpc' "
            f"-H 'te: trailers' --data-binary @- {url} | tail -c +6 | "
            f"COLOSSEUM_HUB={_sq(address)} COLOSSEUM_TOKEN={_sq(str(token))} sh")


def join_command(address: str, token: str, *, insecure: bool) -> str:
    return f"colosseum join {address} --token {_sq(str(token))}" + (" --insecure" if insecure else "")


# ---------------------------------------------------------------------------------------------------------------
# what the hub serves (built once by the hub's main thread at start; immutable)
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class BootstrapService:
    bundle: _bundle.Bundle
    requirements: tuple[str, ...]
    requirements_hash: str
    python: str                   # "3.12"
    torch_version: str            # "2.14.1"
    torch_build: str              # "cu130"
    script: bytes
    protocol: tuple[int, int]
    archive: bytes                # the bundle's bytes (served from memory: pruning the file never breaks a hub)

    def info(self):
        from colosseum.net.proto import colosseum_v1_pb2 as pb

        return pb.BootstrapInfo(protocol_major=self.protocol[0], protocol_minor=self.protocol[1],
                                code_hash=self.bundle.code_hash, requirements_hash=self.requirements_hash,
                                requirements=list(self.requirements), python=self.python,
                                torch_version=self.torch_version, torch_build=self.torch_build,
                                bundle_size=self.bundle.size)


def make_bootstrap_service(*, project_dir: Path, include: Sequence[str], exclude: Sequence[str],
                           max_file_mb: float, requirements: Path | None, run_dir: Path, insecure: bool,
                           out_dir: Path, protocol: tuple[int, int]) -> BootstrapService:
    """Bundle the project, freeze the requirements, write the installer (raises BundleError / ValueError /
    OSError; the hub turns them into a ConfigError)."""
    import importlib.metadata
    import sys

    from colosseum.bootstrap.requirements import node_requirements

    project = Path(project_dir).resolve()
    extra = []
    try:
        rel_run = Path(run_dir).resolve().relative_to(project).as_posix()
        if rel_run not in (".", "runs") and not rel_run.startswith("runs/"):
            extra.append(rel_run)                      # a run dir inside the project is never shipped
    except ValueError:
        pass
    bundle = _bundle.build_bundle(project, include=include, exclude=[*exclude, *extra], max_file_mb=max_file_mb,
                                  out_dir=out_dir)
    reqs, reqs_hash = node_requirements(override=requirements)
    torch_version, torch_build = _torch_select.torch_identity()
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    script = installer_script(insecure=insecure, python=python, grpcio=importlib.metadata.version("grpcio"))
    archive = bundle.path.read_bytes()
    prune_archives(Path(out_dir), keep=KEEP_HUB_BUNDLES, current=bundle.path)
    return BootstrapService(bundle, tuple(reqs), reqs_hash, python, torch_version, torch_build, script, protocol,
                            archive)


def prune_archives(out_dir: Path, *, keep: int, current: Path) -> None:
    """Keep the ``keep`` newest hub archives (by mtime) and ``current``."""
    archives = sorted(out_dir.glob("*" + _bundle.ARCHIVE_SUFFIX), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in archives[keep:]:
        if path != current:
            path.unlink(missing_ok=True)
```

- [ ] **Step 6: The Bootstrap servicer in `hub/server.py`**

Add to `src/colosseum/hub/server.py` (imports merged into the module's import block):
```python
from __future__ import annotations

import grpc

from colosseum.bootstrap.installer import BootstrapService
from colosseum.net.grpc_util import send_parts
from colosseum.net.proto import colosseum_v1_pb2_grpc as pb_grpc

BOOTSTRAP_SERVICE = "colosseum.v1.Bootstrap"


class BootstrapServicer(pb_grpc.BootstrapServicer):
    """``Bootstrap/Info`` (open) and ``Bootstrap/Bundle`` (token secret: ``AuthInterceptor``); read-only, served
    by the servicer threads from the immutable ``BootstrapService`` (spec 2.3)."""

    def __init__(self, service: BootstrapService) -> None:
        self._service = service
        self._info = service.info()

    def Info(self, request, context):  # noqa: N802 (gRPC method name)
        return self._info

    def Bundle(self, request, context):  # noqa: N802
        if request.code_hash != self._service.bundle.code_hash:
            context.abort(grpc.StatusCode.NOT_FOUND, f"code {request.code_hash[:12]} is not served by this hub "
                                                     f"(current: {self._service.bundle.code_hash[:12]})")
        yield from send_parts(self._service.archive)


def add_bootstrap(server: grpc.Server, service: BootstrapService) -> None:
    """Register Info / Bundle (generated) and ``Script`` (raw bytes both ways, R2: the curl one-liner reads the
    script after the 5-byte gRPC prefix) on ``server``."""
    pb_grpc.add_BootstrapServicer_to_server(BootstrapServicer(service), server)
    script = service.script
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(BOOTSTRAP_SERVICE, {
        "Script": grpc.unary_unary_rpc_method_handler(lambda request, context: script,
                                                      request_deserializer=None, response_serializer=None),
    }),))
```
and in `build_server`, after the other servicers are added and before the ports are bound:
```python
    if bootstrap is not None:
        add_bootstrap(server, bootstrap)
```
(`build_server`'s parameter `bootstrap: BootstrapService | None = None` is in the contract; its type now comes from `colosseum.bootstrap.installer`.) The generated `add_BootstrapServicer_to_server` registers Info and Bundle as registered method handlers; the generic handler for the same service name serves only `Script` (grpcio 1.84 tries the generic handlers in order for methods it does not know: verified). `AuthInterceptor` (T1.3) already lets `Bootstrap/Info` and `Bootstrap/Script` through without credentials and requires the token secret for `Bootstrap/Bundle`; if it does not, fix it there (`test_refusals[secret]` and `test_one_liner_end_to_end_with_real_curl` fail otherwise).

- [ ] **Step 7: Run the new tests (GREEN)**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_join_loader.py tests/unit/test_sp5_bootstrap_service.py tests/integration/test_sp5_join_installer.py -q -rw`
Expected: all pass (37 tests, under 5 s); the curl tests run (curl 7.81 with HTTP/2 on the owner's machine). If `test_codec_matches_the_generated_messages` fails on field numbers, T1.2 numbered the Bootstrap messages differently: change `INFO_FIELDS` / `PART_FIELDS` / `BUNDLE_REQUEST_CODE_HASH` to the generated numbers (never the proto).

- [ ] **Step 8: Full fast suite + ruff**

Run the full fast suite and ruff. Expected: green, zero warnings.

- [ ] **Step 9: Commit and push**

```bash
git add src/colosseum/bootstrap/loader.py src/colosseum/bootstrap/installer.py src/colosseum/hub/server.py \
        tests/join_kit.py pyproject.toml tests/unit/test_sp5_join_loader.py tests/unit/test_sp5_bootstrap_service.py \
        tests/integration/test_sp5_join_installer.py
git commit -m "feat: bootstrap service, installer script, curl one-liner and the join loader"
git push
```

---

### Task T6.3: `colosseum join`, registration checks, self-restart, persistent nodes, `nodes list / remove`, `hub token`, join commands, WSL2 hint

Spec 2.1 (`hub token [--rotate]`, nodes kept across runs with the same number and name), 2.2 (the hub prints the join command and the one-liner per address; `nodes list`, `nodes remove NAME [--ban]`), 2.3 (the agent restarts through the loader on a new code hash), 2.5 (registration checks), 2.6 (join flags and slots), 3.5 (WSL2 NAT), 7.5 (`join` exit codes 0 / 1 / 2). Runs after T5.2 (both edit `cli.py`) and T6.2.

**Split point:** dispatch (a) Steps 1–9 (registration checks end to end, the agent's command line and self-restart, `colosseum join`, the hub's bootstrap wiring and start-up messages, the join process test), (b) Steps 10–17 (persistent nodes: run switch, `--once`, release; `nodes list / remove [--ban]`; `hub token`; the two slow tests). (b) starts from (a)'s commit.

**Files:**
- Create: `src/colosseum/net/registration.py`, `src/colosseum/hub/joining.py`
- Modify: `src/colosseum/hub/server.py` (Session servicer: the `Hello` check; `build_server(..., policy=)`; `SessionOpened.warnings`), `src/colosseum/hub/hub.py` (bootstrap service and registration policy at start, start-up messages, `_on_session_opened` warnings, `_on_register` ban check, `_on_admin` `nodes.list` / `nodes.remove`, pending removals, `_node_rows`), `src/colosseum/hub/admin.py` (`format_nodes_table`), `src/colosseum/node/agent.py` (`NodeOptions.code_hash` / `bundle_dir`, `join_slots`, `build_agent_parser`, `options_from_args`, `join_argv`, `main`, `Hello` fields, refusal handling, `_reexec_loader`, `_switch_run`, `--once`, `release`), `src/colosseum/node/session.py` (terminal statuses → node inbox events), `src/colosseum/cli.py` (`join`, `nodes list`, `nodes remove`, `hub token`), `tests/cli_runner.py` (`node_env`, `join_cmd`, `start_join`, `hub_address`; `start_hub(..., cwd=)` if T2.5's lacks `cwd`), `tests/hub_kit.py` / `tests/node_kit.py` only where a fake needs a knob listed below
- Test: `tests/unit/test_sp5_join_checks.py`, `tests/unit/test_sp5_join_hints.py`, `tests/unit/test_sp5_join_slots.py`, `tests/integration/test_sp5_join_hub.py`, `tests/integration/test_sp5_join_node.py`, `tests/integration/test_sp5_join_cli.py`, `tests/integration/test_sp5_join_process.py`, `tests/integration/test_sp5_nodes_admin.py`, `tests/integration/test_sp5_join_persistent_slow.py`, `tests/integration/test_sp5_join_one_liner_slow.py`

**Interfaces:**
- Consumes: T6.2 (`loader_main`, `make_bootstrap_service`, `BootstrapService`, `one_liner`, `join_command`, `add_bootstrap`, `join_kit`); T1.3 (`HubIdentity` incl. `rotate()`, `hub_identity_dir`, `NodeBook.{register, verify, ban, is_banned, remove}`, `NodeIdentity`, `parse_token`, `TOKEN_ENV`, `SecurityError`, `fetch_hub_ca`); T1.2 (`pb.Hello`, `pb.Hardware`, `pb.Command`, `PROTOCOL_VERSION`); T2.3 (Session / Register servicers, `SessionOpened`, `RegisterIn`, `AdminIn`, `HubSession`, `hub_kit.FakeNode`); T2.4 (`NodeAgent`, `NodeOptions`, `Hardware`, `default_slots`, `probe`, `machine_id`, `node_log_dir`, `node_kit.FakeHub`); T2.5 (`Hub`, `HubOptions`, `run_hub`, `hub.json` / `read_hub_json`, `running_hubs`, `register_running`, `admin.resolve_hub`, `admin.call`, the `hub` click group, `cli_runner.start_hub`); T4.x (the node-loss path that drops a node's cells and relocates its learners; the stop sequence of a node: final snapshots, results flush, `shutdown_grace_sec`); T4.3 (the journal's `node` column).
- Produces:
  - `colosseum.net.registration`: `RegistrationPolicy(protocol_major, code_hash, python, torch_version, torch_build, need_bytes)`, `Refusal(reason, message, code_hash="")` with `.details()`, `parse_refusal(details)`, `check_hello(*, os_name, protocol_major, code_hash, python, torch_version, torch_build, shm_free_bytes, cache_free_bytes, policy, source_node) -> (Refusal | None, list[str])`, `need_bytes(weight_bytes)`, `is_loopback_peer(peer)`, `REFUSAL_REASONS`, `CACHE_HEADROOM_BYTES`;
  - `colosseum.hub.joining`: `local_ipv4_addresses()`, `WslNetwork(mode, address, source)`, `wsl_network(*, proc_version, addresses, run)`, `wsl_hint(net, port)`, `hub_addresses(listen, port, *, addresses)`, `join_instructions(addresses, token, *, insecure)`, `redact(lines, token)`, `StartupMessages(console, log, warnings)`, `startup_messages(listen, port, token, *, insecure, addresses=None, wsl=...)`;
  - `colosseum.node.agent`: `NodeOptions.code_hash: str = ""`, `NodeOptions.bundle_dir: Path | None = None`, `NodeAgent(options, *, exec_fn=os.execve)`, `join_slots(hw, *, workers, evaluators, cpu_learners, no_learners, gpus) -> dict`, `build_agent_parser()`, `options_from_args(argv, environ) -> NodeOptions`, `join_argv(options) -> list[str]`, `main(argv=None) -> int` (exit codes of spec 7.5 for `join`);
  - `colosseum.hub.server.build_server(..., policy: Callable[[], RegistrationPolicy] | None = None)`; `SessionOpened.warnings: tuple[str, ...] = ()`;
  - node inbox events (`colosseum.node.session`): `SessionRefused(refusal: Refusal)`, `SessionUnauthenticated(details: str)`, `SessionBanned(details: str)`, `SessionKeyMismatch(message: str)`;
  - admin commands `nodes.list` (reply `data = {"nodes": [row, ...], "banned": int}`) and `nodes.remove` (`args = {"name": str, "ban": bool}`); `Command(kind="release")` to a node; `admin.format_nodes_table(rows) -> str`;
  - CLI: `colosseum join ...` (spec 9.2 flags), `colosseum nodes list`, `colosseum nodes remove NAME [--ban]` (+ `--hub` / `--run` / `--token` / `--token-file`), `colosseum hub token [--rotate]`;
  - `tests/cli_runner.py`: `node_env(tmp_path, name) -> dict`, `join_cmd(address, *, token_file, name, workers=1, evaluators=0, cpu_learners=1, once=True, extra=()) -> list[str]`, `start_join(cmd, env, cwd=None) -> subprocess.Popen` (own session; killed with its group by `stop_session`), `hub_address(run_root, timeout=30.0) -> str`.

**Behaviour (decided here).**

*Registration checks (spec 2.5).* The hub builds one immutable `RegistrationPolicy` at start: `protocol_major = PROTOCOL_VERSION[0]`; `code_hash` = its bundle's hash (`""` without a `BootstrapService`, i.e. plain `train`); `python` = its `major.minor`; `torch_version`, `torch_build` = `torch_select.torch_identity()`; `need_bytes = need_bytes([...])` = 64 MiB + 2 × the weights bytes of every agent the nodes may cache (each trainable agent's network state dict, counted once at start from a model built with the agent's `networks` exactly as the learner builds it; each frozen / teacher / init blob's frame size). The Session servicer reads the first message (`Hello`), calls `check_hello(os_name=hello.hardware.os, protocol_major=hello.protocol_major, code_hash=hello.code_hash, python=hello.hardware.python, torch_version=hello.hardware.torch_version, torch_build=hello.hardware.torch_build, shm_free_bytes=hello.hardware.shm_free_bytes, cache_free_bytes=hello.hardware.cache_free_bytes, policy=policy(), source_node=is_loopback_peer(context.peer()))` and, on a refusal, `context.abort(grpc.StatusCode.FAILED_PRECONDITION, refusal.details())` before any `Welcome` (no inbox event besides a `RelayEvent`-style counter is needed; it logs one WARNING per (node id, reason) itself: logging is thread-safe). Accepted: `SessionOpened(handle, hello, warnings=tuple(warnings))`; `Hub._on_session_opened` logs each warning once per (node, text). A `train` hub (`code_hash == ""`) accepts only source nodes (empty code hash from loopback): its local node (R12) and in-process test nodes.

*Node side of a refusal.* `HubSession` (T2.3) classifies how a session RPC ends: `FAILED_PRECONDITION` whose details parse with `parse_refusal` → posts `SessionRefused(refusal)` and stops reconnecting; `UNAUTHENTICATED` → `SessionUnauthenticated(details)`, stops; `PERMISSION_DENIED` → `SessionBanned(details)`, stops; a `security.SecurityError` from the pin check (the hub's key is no longer the token's, e.g. after `hub token --rotate`) → `SessionKeyMismatch(message)`, stops; everything else stays a network error with backoff (T2.3 / T4.4). `NodeAgent` handles them in its main loop:
- `SessionRefused` with reason `code_hash` or `protocol` and a `refusal.code_hash` different from `options.code_hash`: when `options.bundle_dir` is set (a node started by the loader) → `_reexec_loader()`; else (a source node) → ERROR "this node runs source code but the hub serves code <hash>: start it with `colosseum join`", exit 1;
- any other `SessionRefused` → ERROR with `refusal.message`, exit 1;
- `SessionUnauthenticated` (unknown node id or secret: the record was removed, `nodes.json` lost) → once per process: register again with the token (`RegisterRequest(node_id=<stored id>, ...)`), store the new registration, reconnect; a second one, or `UNAUTHENTICATED` from `Register` (the token secret is refused) → ERROR "the hub refused the token (colosseum hub token prints the current one)", exit 1;
- `SessionBanned`, or `PERMISSION_DENIED` from `Register` → ERROR "this machine is banned by the hub (colosseum nodes remove --ban); `colosseum hub token --rotate` on the hub resets every node", exit 1;
- `SessionKeyMismatch` → ERROR with the pin message, exit 1.

*Self-restart (spec 2.3).* `NodeAgent._reexec_loader()`: stop the children with the node's stop sequence (T4.4: workers and evaluators stopped, results flushed while the session allows, learners' final snapshots — none can be uploaded after a refusal, so they are skipped; bounded by `distributed.shutdown_grace_sec`), close the session, log INFO `code changed (<old12> -> <new12>): restarting through the loader`, then `self._exec(sys.executable, [sys.executable, "-I", str(Path(colosseum.bootstrap.loader.__file__)), *join_argv(self.options)], {**os.environ, TOKEN_ENV: str(self.options.token)})`. The loader file is the bundle's own copy (the agent runs from the bundle); the agent's environment `python` has grpcio; `-I` drops the old bundle's `PYTHONPATH`. T8.3 calls the same method for `Command(kind="update_code")`.

*The agent's command line.* `main(argv=None)`: `options = options_from_args(argv or sys.argv[1:], os.environ)` (code below; `ValueError` / `SecurityError` → the message on stderr, exit 2); `os.environ.pop(TOKEN_ENV)` right after (children never see the token; the self-restart passes it explicitly); `hw = hardware.probe()`; `slots = join_slots(hw, ...)` (`ValueError` → exit 2); then `NodeAgent(options).run()` with the signal handling of T4.4, returning its exit code. The default name is the short host name (`socket.gethostname().split(".")[0]`; `NodeBook` adds `-2`, `-3`). The agent logs at start: `node agent: code <hash12> from <bundle dir>` (or `source tree <colosseum dir>`), and on the first `Welcome` of a run `joined run <run_id> as node <number> (<name>): <workers> workers, <evaluators> evaluators, learner slots <cpu> CPU + GPUs <list>`.

*`Hello` fields (T2.4 fills most; T6.3 makes sure):* `code_hash = options.code_hash`; `hardware.os = platform.system()`; `hardware.python = platform.python_version()`; `hardware.torch_version, hardware.torch_build = torch_select.torch_identity()`; `hardware.shm_free_bytes = statvfs(shm_dir())` free bytes; `hardware.cache_free_bytes = statvfs(cache_home())` free bytes (0 when the directory cannot be inspected: the hub then refuses with reason `shm` / `disk` and its hint).

*Bootstrap wiring and start-up messages (spec 2.2, 3.5).* In modes `hub` and `idle`, and in `train` with `--listen`, `Hub` builds the `BootstrapService` before `build_server`: `make_bootstrap_service(project_dir=Path.cwd(), include=cfg.distributed.bundle.include, exclude=cfg.distributed.bundle.exclude, max_file_mb=cfg.distributed.bundle.max_file_mb, requirements=Path(cfg.distributed.requirements) if set else None, run_dir=Path(cfg.run.dir), insecure=options.insecure, out_dir=cache_home() / "colosseum" / "hub-bundles", protocol=PROTOCOL_VERSION)`; `BundleError`, `ValueError`, `OSError` (a missing requirements file) become `ConfigError` with the message (exit 1 before any child starts); it logs INFO `code bundle <hash12>: <files> files, <MB> MB; <n> requirements`. Plain `train` passes `bootstrap=None`. After the external port is bound: `msgs = startup_messages(options.listen, bound_port, str(identity.token), insecure=options.insecure)`; `msgs.console` is printed to stdout (`print(..., flush=True)`; never logged: it carries the token), `msgs.log` is logged at INFO, each of `msgs.warnings` logged at WARNING (console and `main.log`).

*Persistent nodes (spec 2.1, 2.2).* A node stays after its run: on `Command(kind="stop_run")` it runs the stop sequence (T4.4), keeps its identity and reconnects with the session backoff; with `--once` it exits 0 once the stop sequence finished. `Welcome.run_id` different from the run the node worked for (a new run of the same hub identity) → `_switch_run(welcome)`: children of the old run stopped if any still run; unacked result batches of the old run dropped and counted (one WARNING `N result batches of run <old> were never acknowledged (the run ended): dropped`); cells, epochs, routes and per-run counters reset; the old run's shm prefix (`cache_prefix(ca, old_run, number)`) removed; logs move to `node_log_dir(ca8, new_run_id)`; then as on a first `Welcome`. (The same `run_id` after a hub `--resume` is T4.4's resume path: buffers kept and resent.) A code-hash refusal of the next run restarts through the loader first (above).

*`nodes remove NAME [--ban]` (spec 2.2).* `Hub._on_admin("nodes.remove", {"name", "ban"})`: the `NodeBook` record with that name (exact match; none → reply `code 1`, "no node named X (colosseum nodes list)"). Online → send `Command(kind="release", reason="removed by colosseum nodes remove")`, treat the node at once like a lost node for placement and cells (its cells dropped, its learners relocated without waiting `learner_relocate_after_sec`: the T4.2 path, with a reason `removed`), and keep the admin reply pending until its session closes or `distributed.shutdown_grace_sec + 5 s` pass on the hub clock; then `NodeBook.ban(node_id)` (with `--ban`; the record's `machine_id` is banned too) or `NodeBook.remove(node_id)`, and reply `code 0`, `node 3 (laptop) released` (+ ` and banned`; + `; it did not stop within <s> s` when the deadline passed). The book is changed only after the session closed, so the released node can still upload its learners' final snapshots (`Upload` is authenticated per call). Offline → the book change and the reply at once. The node on `release`: the stop sequence (final snapshots, result flush, `shutdown_grace_sec`), then `run()` returns 0 (with or without `--once`). Unfinished episodes are not played to their end (proposed ruling 4).

*`nodes list`.* `Hub._node_rows()` (T7.1 writes `nodes.json` from the same rows): for every `NodeBook` record plus every registry entry, ordered by number: `{"number", "name", "node_id", "state": "online" | "lost" | "offline", "hardware": {"logical_cores", "physical_cores", "memory_bytes", "gpus": [{"index", "name", "memory_bytes"}], "os", "torch": "<version>+<build>"}, "roles": {"workers", "evaluators", "learners": ["<agent>@<device>"]}, "data_path": "direct" | "relay" | "local" | None, "last_heartbeat_age_sec": float | None, "tags": [...], "code_hash": str}` (`offline` = registered, not connected to this hub process; its hardware from the last `Hello` of this process, else `{}`). Reply `data = {"nodes": rows, "banned": len(banned)}`. `admin.format_nodes_table(rows)` prints `#  NAME  STATE  CPU  MEM  GPUS  WORKERS  EVAL  LEARNERS  PATH  SEEN  TAGS` (CPU = physical/logical, MEM in GiB, SEEN in seconds or `-`) and a last line `banned: N` when N > 0.

*`hub token [--rotate]`.* `colosseum hub token` loads or creates the identity of `hub_identity_dir()` and prints `str(identity.token)` alone on stdout (scriptable: `colosseum hub token > token.txt`). `--rotate`: `running_hubs(identity_dir)` not empty → stderr `a hub of this identity is running (pid P, run R): stop it first`, exit 1; else `identity.rotate()`, print the new token, and on stderr `every node is revoked: join the machines again with the new token`. The `hub` group callback (T2.5) must not require `-c` when a subcommand is invoked (`ctx.invoked_subcommand`).

*`colosseum join`.* A click command with the spec 9.2 flags (help text below) that validates syntax (exit 2: `--token` with `--token-file`, a non-integer or negative count, a malformed `--gpus`), rebuilds the loader's argv from the parsed values and returns `loader_main(argv)` as its exit code. It imports neither torch nor colosseum's training modules.

- [ ] **Step 1: (a) Write the failing unit tests**

Create `tests/unit/test_sp5_join_checks.py`:
```python
"""Registration checks of a node's Hello (SP5 T6.3, spec 2.5): each refusal, the build warning, source nodes,
refusal details on the wire, loopback peers."""
from __future__ import annotations

import pytest

from colosseum.net import registration as J

POLICY = J.RegistrationPolicy(protocol_major=1, code_hash="c" * 64, python="3.12", torch_version="2.14.1",
                              torch_build="cu130", need_bytes=J.need_bytes([10 << 20, 30 << 20]))
GOOD = dict(os_name="Linux", protocol_major=1, code_hash="c" * 64, python="3.12.15", torch_version="2.14.1",
            torch_build="cu130", shm_free_bytes=1 << 30, cache_free_bytes=1 << 34)


def test_need_bytes():
    assert POLICY.need_bytes == (64 << 20) + 2 * (40 << 20)


@pytest.mark.parametrize("change, reason", [
    ({"os_name": "Windows"}, "platform"),
    ({"protocol_major": 2}, "protocol"),
    ({"code_hash": "d" * 64}, "code_hash"),
    ({"code_hash": ""}, "code_hash"),
    ({"python": "3.11.9"}, "python"),
    ({"torch_version": "2.13.0"}, "torch"),
    ({"shm_free_bytes": 100 << 20}, "shm"),
    ({"cache_free_bytes": 100 << 20}, "disk"),
])
def test_each_refusal(change, reason):
    refusal, warnings = J.check_hello(**{**GOOD, **change}, policy=POLICY, source_node=False)
    assert refusal is not None and refusal.reason == reason and warnings == []
    assert J.parse_refusal(refusal.details()) == refusal
    if reason == "code_hash" and change["code_hash"]:
        assert refusal.code_hash == "c" * 64


def test_accepted_with_build_warning_and_source_nodes():
    assert J.check_hello(**GOOD, policy=POLICY, source_node=False) == (None, [])
    refusal, warnings = J.check_hello(**{**GOOD, "torch_build": "cpu"}, policy=POLICY, source_node=False)
    assert refusal is None and warnings == ["torch build cpu on the node, the hub runs cu130"]
    assert J.check_hello(**{**GOOD, "code_hash": ""}, policy=POLICY, source_node=True) == (None, [])
    assert J.check_hello(**{**GOOD, "code_hash": "d" * 64}, policy=POLICY, source_node=True)[0].reason == "code_hash"
    plain = J.RegistrationPolicy(1, "", "3.12", "2.14.1", "cu130", 0)
    assert J.check_hello(**{**GOOD, "code_hash": ""}, policy=plain, source_node=True) == (None, [])
    refusal = J.check_hello(**{**GOOD, "code_hash": "c" * 64}, policy=plain, source_node=False)[0]
    assert refusal.reason == "code_hash" and "colosseum train --listen" in refusal.message
    assert J.parse_refusal("not json") is None and J.parse_refusal('{"refused": "other"}') is None


@pytest.mark.parametrize("peer, loop", [("ipv4:127.0.0.1:5555", True), ("ipv6:[::1]:5555", True),
                                        ("unix:/tmp/x", True), ("ipv4:10.0.0.7:5555", False),
                                        ("ipv6:[2001:db8::1]:1", False), ("garbage", False)])
def test_loopback_peer(peer, loop):
    assert J.is_loopback_peer(peer) is loop
```

Create `tests/unit/test_sp5_join_hints.py`:
```python
"""Hub-side joining helpers (SP5 T6.3, spec 2.2, 3.5): addresses, join instructions (token only on the console),
WSL2 NAT detection and the portproxy hint."""
from __future__ import annotations

import subprocess

from colosseum.hub import joining as H


def test_addresses_and_instructions():
    assert H.hub_addresses("192.168.1.5", 47100, addresses=["10.0.0.1"]) == ["192.168.1.5:47100"]
    assert H.hub_addresses(None, 47100, addresses=["10.0.0.1", "172.31.1.2"]) == ["10.0.0.1:47100", "172.31.1.2:47100"]
    assert H.hub_addresses("0.0.0.0", 1, addresses=[]) == ["127.0.0.1:1"]
    token = "cst1." + "S" * 43 + "." + "f" * 64
    lines = H.join_instructions(["10.0.0.1:47100"], token, insecure=False)
    assert f"  colosseum join 10.0.0.1:47100 --token '{token}'" in lines
    assert any("curl" in line and "--pinnedpubkey" in line for line in lines)
    assert all("S" * 43 not in line for line in H.redact(lines, token))
    assert all("127.0.0.1" not in a for a in H.local_ipv4_addresses())


def _wslinfo(stdout, rc=0):
    def run(cmd, **kw):
        if stdout is None:
            raise FileNotFoundError(cmd[0])
        return subprocess.CompletedProcess(cmd, rc, stdout, "")
    return run


WSL = "Linux version 6.18.54.1-microsoft-standard-WSL2 (root@x) #1 SMP"


def test_wsl_detection():
    assert H.wsl_network(proc_version="Linux version 6.8.0-45-generic", addresses=[]) is None
    net = H.wsl_network(proc_version=WSL, addresses=["172.31.101.51", "192.168.127.2"], run=_wslinfo("nat\n"))
    assert net == H.WslNetwork("nat", "172.31.101.51", "wslinfo")
    assert H.wsl_network(proc_version=WSL, addresses=["172.31.1.2"], run=_wslinfo(None)).mode == "nat"
    assert H.wsl_network(proc_version=WSL, addresses=["192.168.1.9"], run=_wslinfo(None)).mode == "unknown"
    mirrored = H.wsl_network(proc_version=WSL, addresses=["192.168.1.9"], run=_wslinfo("mirrored"))
    assert H.wsl_hint(mirrored, 47100) is None
    hint = "\n".join(H.wsl_hint(net, 47100))
    assert ("netsh interface portproxy add v4tov4 listenport=47100 listenaddress=0.0.0.0 connectport=47100 "
            "connectaddress=172.31.101.51") in hint
    assert 'New-NetFirewallRule -DisplayName "Colosseum hub 47100"' in hint and "DISTRIBUTED_GUIDE" in hint


def test_startup_messages():
    token = "cst1." + "S" * 43 + "." + "f" * 64
    nat = H.WslNetwork("nat", "172.31.1.2", "wslinfo")
    m = H.startup_messages(None, 47100, token, insecure=False, addresses=["172.31.1.2"], wsl=nat)
    assert m.console[0] == "Hub listening on all interfaces, port 47100 (TLS)."
    assert any(token in line for line in m.console) and not any("S" * 43 in line for line in m.log)
    assert any("portproxy" in w for w in m.warnings)
    m = H.startup_messages("127.0.0.1", 5000, token, insecure=True, addresses=[], wsl=nat)
    assert m.console[2] == f"  colosseum join 127.0.0.1:5000 --token '{token}' --insecure"
    assert not any("portproxy" in w for w in m.warnings) and "clear text" in m.warnings[0]
    assert H.startup_messages(None, 1, token, insecure=False, addresses=[], wsl=None).warnings == []
```

Create `tests/unit/test_sp5_join_slots.py` (the `Hardware` fields and `default_slots` rule are T2.4's contract):
```python
"""Join flags of a node (SP5 T6.3, spec 2.6): slots from the hardware defaults and the flags; the agent's command
line (the loader's arguments) to NodeOptions and back."""
from __future__ import annotations

import pytest

from colosseum.net.security import SecurityError
from colosseum.node.agent import join_argv, join_slots, options_from_args
from colosseum.node.hardware import Hardware

TOKEN = "cst1." + "S" * 43 + "." + "f" * 64


def hw(physical=10, gpus=1):
    return Hardware(logical_cores=2 * physical, physical_cores=physical, memory_bytes=32 << 30,
                    gpus=tuple({"index": i, "name": "RTX", "memory_bytes": 12 << 30, "kind": "cuda"}
                               for i in range(gpus)),
                    os="Linux", platform="linux-x86_64", python="3.12.15", torch_version="2.14.1", torch_build="cu130")


def slots(h, **kw):
    kw = {"workers": None, "evaluators": None, "cpu_learners": None, "no_learners": False, "gpus": None, **kw}
    return join_slots(h, **kw)


def test_defaults_follow_the_hardware():
    s = slots(hw(10, 1))
    assert (s["workers"], s["evaluators"], s["cpu_learners"], s["gpus"]) == (9, 1, 1, [0])
    s = slots(hw(2, 0))
    assert (s["workers"], s["evaluators"], s["cpu_learners"], s["gpus"]) == (1, 0, 0, [])


def test_flags_override():
    s = slots(hw(10, 2), workers=0, evaluators=3, cpu_learners=2, gpus=[1])
    assert (s["workers"], s["evaluators"], s["cpu_learners"], s["gpus"]) == (0, 3, 2, [1])
    s = slots(hw(10, 2), no_learners=True)
    assert (s["cpu_learners"], s["gpus"], s["no_learners"]) == (0, [], True)
    with pytest.raises(ValueError, match="no GPU 2 on this machine"):
        slots(hw(10, 2), gpus=[2])
    with pytest.raises(ValueError, match="--no-learners"):
        slots(hw(10, 2), no_learners=True, cpu_learners=1)


def test_agent_argv_round_trip():
    argv = ["10.0.0.5:47100", "--workers", "3", "--gpus", "0,1", "--name", "lap", "--tags", "gpu,big", "--once",
            "--insecure", "--code-hash", "c" * 64, "--bundle", "/b"]
    opts = options_from_args(argv, {"COLOSSEUM_TOKEN": TOKEN})
    assert (opts.workers, opts.gpus, opts.name, opts.tags, opts.once, opts.insecure) == \
        (3, (0, 1), "lap", ("gpu", "big"), True, True)
    assert (opts.code_hash, str(opts.bundle_dir), str(opts.token)) == ("c" * 64, "/b", TOKEN)
    assert join_argv(opts) == ["10.0.0.5:47100", "--workers", "3", "--name", "lap", "--gpus", "0,1", "--tags",
                               "gpu,big", "--once", "--insecure"]


@pytest.mark.parametrize("argv, env, error", [
    (["h:1", "--gpus", "a"], {"COLOSSEUM_TOKEN": TOKEN}, ValueError),
    (["h:1", "--workers", "-1"], {"COLOSSEUM_TOKEN": TOKEN}, ValueError),
    (["h:1", "--code-hash", "xyz"], {"COLOSSEUM_TOKEN": TOKEN}, ValueError),
    (["h:1"], {}, SecurityError),
    (["h:1"], {"COLOSSEUM_TOKEN": "cst1.bad"}, SecurityError),
])
def test_bad_agent_arguments(argv, env, error):
    with pytest.raises(error):
        options_from_args(argv, env)
```

Run them: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_join_checks.py tests/unit/test_sp5_join_hints.py tests/unit/test_sp5_join_slots.py -q`
Expected (RED): `ModuleNotFoundError: No module named 'colosseum.net.registration'`, `... 'colosseum.hub.joining'`, `ImportError: cannot import name 'join_argv' from 'colosseum.node.agent'`.

- [ ] **Step 2: (a) Write `registration.py` and `hub/joining.py`**

`src/colosseum/net/registration.py`:
```python
"""Registration checks of a node's ``Hello`` (SP5 spec 2.5), shared by the hub (Session servicer) and the node
(which reads a refusal from the session's gRPC status). Pure functions, standard library only."""

from __future__ import annotations

import ipaddress
import json
from collections.abc import Sequence
from dataclasses import dataclass

MIB = 1 << 20
CACHE_HEADROOM_BYTES = 64 * MIB
REFUSAL_REASONS = ("platform", "protocol", "code_hash", "python", "torch", "shm", "disk")


@dataclass(frozen=True)
class RegistrationPolicy:
    """What the hub requires of a node at ``Hello`` (immutable; swapped whole when the hub's code changes)."""

    protocol_major: int
    code_hash: str                # "" when the hub serves no bundle (plain ``train``)
    python: str                   # "3.12"
    torch_version: str            # "2.14.1"
    torch_build: str              # "cu130"
    need_bytes: int               # expected weights in a node's cache (shm and cache disk)


@dataclass(frozen=True)
class Refusal:
    reason: str                   # one of REFUSAL_REASONS
    message: str                  # for the node's terminal and the hub log
    code_hash: str = ""           # the hub's code hash (reason "code_hash": the node fetches it and restarts)

    def details(self) -> str:
        """The gRPC status details the hub aborts the session with (FAILED_PRECONDITION)."""
        return json.dumps({"refused": self.reason, "message": self.message, "code_hash": self.code_hash})


def parse_refusal(details: str | None) -> Refusal | None:
    try:
        data = json.loads(details or "")
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("refused") not in REFUSAL_REASONS:
        return None
    return Refusal(data["refused"], str(data.get("message", "")), str(data.get("code_hash", "")))


def _major_minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


def need_bytes(weight_bytes: Sequence[int]) -> int:
    """Expected cache use: two versions of every agent's weights plus 64 MiB (the cache's own write rule)."""
    return CACHE_HEADROOM_BYTES + 2 * sum(weight_bytes)


def check_hello(*, os_name: str, protocol_major: int, code_hash: str, python: str, torch_version: str,
                torch_build: str, shm_free_bytes: int, cache_free_bytes: int, policy: RegistrationPolicy,
                source_node: bool) -> tuple[Refusal | None, list[str]]:
    """(refusal or None, warnings) for one ``Hello`` (spec 2.5). ``source_node``: the session comes from this
    machine's loopback and carries no code hash (``train``'s local node, nodes started in-process): it runs the
    hub's own source tree, so the code hash is not checked."""
    if os_name == "Windows":
        return Refusal("platform", "native Windows is not supported: run the node inside WSL2"), []
    if protocol_major != policy.protocol_major:
        return Refusal("protocol", f"protocol {protocol_major}.x, the hub speaks {policy.protocol_major}.x",
                       policy.code_hash), []
    if not (source_node and not code_hash) and code_hash != policy.code_hash:
        if not policy.code_hash:
            return Refusal("code_hash", "this hub serves no code to other machines (start `colosseum hub` or "
                                        "`colosseum train --listen`)"), []
        return Refusal("code_hash", f"code {code_hash[:12] or '(none)'} differs from the hub's "
                                    f"{policy.code_hash[:12]}: fetching the hub's code", policy.code_hash), []
    if _major_minor(python) != policy.python:
        return Refusal("python", f"Python {_major_minor(python)} on the node, the hub runs {policy.python} "
                                 "(do not set COLOSSEUM_NODE_PYTHON to another version)"), []
    if torch_version != policy.torch_version:
        return Refusal("torch", f"torch {torch_version} on the node, the hub runs {policy.torch_version}"), []
    if shm_free_bytes < policy.need_bytes:
        return Refusal("shm", f"{shm_free_bytes >> 20} MiB free in the shared-memory dir, the run needs about "
                              f"{policy.need_bytes >> 20} MiB (docker: --shm-size; Kubernetes: an emptyDir with "
                              "medium: Memory)"), []
    if cache_free_bytes < policy.need_bytes:
        return Refusal("disk", f"{cache_free_bytes >> 20} MiB free on the cache disk, the run needs about "
                               f"{policy.need_bytes >> 20} MiB (free space under ~/.cache/colosseum)"), []
    warnings = []
    if torch_build != policy.torch_build:
        warnings.append(f"torch build {torch_build} on the node, the hub runs {policy.torch_build}")
    return None, warnings


def is_loopback_peer(peer: str) -> bool:
    """gRPC ``context.peer()`` of a client on this machine (``ipv4:127.0.0.1:N``, ``ipv6:[::1]:N``, ``unix:``)."""
    if peer.startswith("unix:"):
        return True
    host = peer.split(":", 1)[1] if ":" in peer else peer
    host = host.rsplit(":", 1)[0].strip("[]")
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
```

`src/colosseum/hub/joining.py`:
```python
"""Joining machines, hub side (SP5 spec 2.2, 3.5): the hub's addresses, the join commands it prints and the WSL2
NAT warning. Pure functions plus two probes with injectable inputs (``local_ipv4_addresses``, ``wsl_network``)."""

from __future__ import annotations

import ipaddress
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from colosseum.bootstrap.installer import join_command, one_liner

WSL_NAT_NET = ipaddress.ip_network("172.16.0.0/12")
REACHABLE_WSL_MODES = ("mirrored", "bridged")
_UNSET = object()


def local_ipv4_addresses() -> list[str]:
    """Non-loopback IPv4 addresses of this machine's interfaces (Linux: ``SIOCGIFADDR`` per interface; elsewhere
    the hostname's addresses), in interface order."""
    found: list[str] = []
    if sys.platform.startswith("linux"):
        import fcntl
        import struct

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            for _, name in socket.if_nameindex():
                try:
                    raw = fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", name[:15].encode()))  # SIOCGIFADDR
                except OSError:
                    continue
                addr = socket.inet_ntoa(raw[20:24])
                if not ipaddress.ip_address(addr).is_loopback and addr not in found:
                    found.append(addr)
    else:
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                addr = info[4][0]
                if not ipaddress.ip_address(addr).is_loopback and addr not in found:
                    found.append(addr)
        except OSError:
            pass
    return found


@dataclass(frozen=True)
class WslNetwork:
    mode: str                     # "nat", "mirrored", "consomme", ... or "unknown"
    address: str | None           # the WSL VM's address (portproxy target)
    source: str                   # "wslinfo" or "address"


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def wsl_network(*, proc_version: str | None = None, addresses: Sequence[str] | None = None,
                run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> WslNetwork | None:
    """The WSL2 networking of this machine; None outside WSL2. Mode from ``wslinfo --networking-mode`` (WSL
    2.0.4+), else ``nat`` when an interface address is in 172.16.0.0/12, else ``unknown`` (spec 3.5)."""
    text = (proc_version if proc_version is not None else _read("/proc/version")).lower()
    if "microsoft" not in text or ("wsl2" not in text and "microsoft-standard" not in text):
        return None
    addrs = list(addresses if addresses is not None else local_ipv4_addresses())
    nat_addr = next((a for a in addrs if ipaddress.ip_address(a) in WSL_NAT_NET), None)
    try:
        out = run(["wslinfo", "--networking-mode"], capture_output=True, text=True, timeout=5)
        mode = out.stdout.strip().lower() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        mode = ""
    if mode:
        return WslNetwork(mode, nat_addr or (addrs[0] if addrs else None), "wslinfo")
    return WslNetwork("nat" if nat_addr else "unknown", nat_addr, "address")


def wsl_hint(net: WslNetwork, port: int) -> list[str] | None:
    """The WARNING lines a hub in WSL2 prints when the LAN cannot reach it (None when it can)."""
    if net.mode in REACHABLE_WSL_MODES:
        return None
    target = net.address or "<WSL address: wsl.exe hostname -I>"
    return [
        f"This hub runs in WSL2 with {net.mode} networking: other machines cannot reach port {port} until the "
        "Windows host forwards it. On the Windows host, in an administrator PowerShell:",
        f"  netsh interface portproxy add v4tov4 listenport={port} listenaddress=0.0.0.0 connectport={port} "
        f"connectaddress={target}",
        f"  New-NetFirewallRule -DisplayName \"Colosseum hub {port}\" -Direction Inbound -Action Allow "
        f"-Protocol TCP -LocalPort {port}",
        "  (the WSL address changes when WSL restarts: run the netsh line again). Nodes then join with the Windows "
        "host's LAN address instead of the addresses above.",
        "  Windows 11 22H2+: networkingMode=mirrored in %UserProfile%\\.wslconfig plus a Hyper-V firewall rule; "
        "or Tailscale inside WSL. Details: docs/DISTRIBUTED_GUIDE.md (WSL2).",
    ]


def hub_addresses(listen: str | None, port: int, *, addresses: Sequence[str] | None = None) -> list[str]:
    """``host:port`` lines for the join commands: the ``--listen`` address, else every non-loopback IPv4 address
    (127.0.0.1 when there is none)."""
    if listen and listen not in ("0.0.0.0", "::", "[::]"):
        host = listen.rsplit(":", 1)[0] if listen.count(":") == 1 else listen
        return [f"{host}:{port}"]
    addrs = list(addresses if addresses is not None else local_ipv4_addresses()) or ["127.0.0.1"]
    return [f"{a}:{port}" for a in addrs]


def join_instructions(addresses: Sequence[str], token: str, *, insecure: bool) -> list[str]:
    """The lines the hub prints to its console (never to a log file: they carry the token)."""
    lines = ["Join a machine with colosseum installed:"]
    lines += [f"  {join_command(a, token, insecure=insecure)}" for a in addresses]
    lines += ["  (or: colosseum hub token > token.txt, then colosseum join ADDRESS --token-file token.txt)",
              "Join a machine without colosseum (Linux, macOS, WSL2; needs curl):"]
    lines += [f"  {one_liner(a, token, insecure=insecure)}" for a in addresses]
    return lines


def redact(lines: Sequence[str], token: str) -> list[str]:
    secret = token.split(".")[1] if token.count(".") == 2 else token
    return [line.replace(secret, "<secret: colosseum hub token>") for line in lines]


@dataclass
class StartupMessages:
    console: list[str] = field(default_factory=list)    # printed to stdout only (they carry the token)
    log: list[str] = field(default_factory=list)        # the same, redacted, for main.log
    warnings: list[str] = field(default_factory=list)   # WARNING lines (console and log)


def _loopback_only(listen: str | None) -> bool:
    if not listen:
        return False
    host = listen.rsplit(":", 1)[0] if listen.count(":") == 1 else listen.strip("[]")
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def startup_messages(listen: str | None, port: int, token: str, *, insecure: bool,
                     addresses: Sequence[str] | None = None, wsl: object = _UNSET) -> StartupMessages:
    """What a joinable hub (``hub``, ``train --listen``) prints after it has bound ``port``."""
    mode = "insecure: no TLS" if insecure else "TLS"
    head = f"Hub listening on {listen or 'all interfaces'}, port {port} ({mode})."
    console = [head, *join_instructions(hub_addresses(listen, port, addresses=addresses), token, insecure=insecure)]
    out = StartupMessages(console, redact(console, token))
    if insecure:
        out.warnings.append("--insecure: the token secret and node keys travel in clear text and data can be "
                            "altered on the way; use it only on a trusted network")
    if not _loopback_only(listen):
        net = wsl_network() if wsl is _UNSET else wsl
        hint = wsl_hint(net, port) if net is not None else None
        if hint:
            out.warnings.extend(hint)
    return out
```

- [ ] **Step 3: (a) The agent's join functions in `node/agent.py`**

Add `code_hash: str = ""` and `bundle_dir: Path | None = None` to `NodeOptions` (after `data_port`), and these functions (module level; `NodeOptions`, `Hardware`, `default_slots`, `TOKEN_ENV`, `SecurityError`, `parse_token` imported as the module already does or adds):
```python
# imports node/agent.py needs for these (merge into its import block):
import argparse
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from colosseum.net.security import TOKEN_ENV, SecurityError, parse_token
from colosseum.node.hardware import Hardware, default_slots


def join_slots(hw: Hardware, *, workers: int | None, evaluators: int | None, cpu_learners: int | None,
               no_learners: bool, gpus: Sequence[int] | None) -> dict:
    """Slots of a ``join`` node (spec 2.6): ``default_slots(hw)`` (workers = physical cores - 1, evaluators and CPU
    learners 1 on >= 4 physical cores, one learner slot per GPU) overridden by the join flags. ``ValueError`` for an
    unknown GPU index or ``--no-learners`` combined with learner flags (the agent exits 2)."""
    slots = dict(default_slots(hw))
    present = sorted(int(g["index"]) for g in hw.gpus)
    if no_learners and (cpu_learners or gpus):
        raise ValueError("--no-learners cannot be combined with --cpu-learners or --gpus")
    if gpus is not None:
        unknown = sorted(set(gpus) - set(present))
        if unknown:
            raise ValueError(f"--gpus: no GPU {unknown[0]} on this machine (GPUs: "
                             f"{', '.join(map(str, present)) or 'none'})")
        slots["gpus"] = sorted(set(gpus))
    if workers is not None:
        slots["workers"] = workers
    if evaluators is not None:
        slots["evaluators"] = evaluators
    if cpu_learners is not None:
        slots["cpu_learners"] = cpu_learners
    if no_learners:
        slots.update(cpu_learners=0, gpus=[], no_learners=True)
    return slots


def build_agent_parser() -> argparse.ArgumentParser:
    """The node agent's command line: the loader's ``agent_args`` plus ``--code-hash`` and ``--bundle``."""
    p = argparse.ArgumentParser(prog="python -m colosseum.node.agent")
    p.add_argument("address")
    p.add_argument("--workers", type=int)
    p.add_argument("--evaluators", type=int)
    p.add_argument("--cpu-learners", type=int)
    p.add_argument("--gpus")
    p.add_argument("--name")
    p.add_argument("--tags")
    p.add_argument("--no-learners", action="store_true")
    p.add_argument("--once", action="store_true")
    p.add_argument("--insecure", action="store_true")
    p.add_argument("--code-hash", default="")
    p.add_argument("--bundle", default=None)
    return p


def options_from_args(argv: Sequence[str], environ: Mapping[str, str]) -> NodeOptions:
    """NodeOptions of a ``join`` node; ``ValueError`` for a bad value, ``SecurityError`` for a missing or malformed
    token (both: exit 2 in ``main``)."""
    ns = build_agent_parser().parse_args(list(argv))
    for flag in ("workers", "evaluators", "cpu_learners"):
        if getattr(ns, flag) is not None and getattr(ns, flag) < 0:
            raise ValueError(f"--{flag.replace('_', '-')} must be >= 0")
    if ns.gpus is not None and not re.fullmatch(r"\d+(,\d+)*", ns.gpus):
        raise ValueError("--gpus: a comma-separated list of GPU indices, e.g. 0,1")
    text = environ.get(TOKEN_ENV)
    if not text:
        raise SecurityError("no token in COLOSSEUM_TOKEN (the loader passes it)")
    if ns.code_hash and not re.fullmatch(r"[0-9a-f]{64}", ns.code_hash):
        raise ValueError("--code-hash: 64 hex characters")
    return NodeOptions(
        hub_address=ns.address, token=parse_token(text), insecure=ns.insecure,
        workers=ns.workers, evaluators=ns.evaluators, cpu_learners=ns.cpu_learners, no_learners=ns.no_learners,
        gpus=tuple(int(g) for g in ns.gpus.split(",")) if ns.gpus else None,
        name=ns.name, tags=tuple(t for t in (ns.tags or "").split(",") if t), once=ns.once,
        code_hash=ns.code_hash, bundle_dir=Path(ns.bundle) if ns.bundle else None)


def join_argv(options: NodeOptions) -> list[str]:
    """The loader's arguments that reproduce ``options`` (the self-restart on a new code hash); no token."""
    out = [options.hub_address]
    for flag, value in (("--workers", options.workers), ("--evaluators", options.evaluators),
                        ("--cpu-learners", options.cpu_learners), ("--name", options.name)):
        if value is not None:
            out += [flag, str(value)]
    if options.gpus is not None:
        out += ["--gpus", ",".join(map(str, options.gpus))]
    if options.tags:
        out += ["--tags", ",".join(options.tags)]
    out += [flag for flag, on in (("--no-learners", options.no_learners), ("--once", options.once),
                                  ("--insecure", options.insecure)) if on]
    return out
``` Run the three unit test files: GREEN.

- [ ] **Step 4: (a) Write the failing integration tests of the registration path**

`tests/integration/test_sp5_join_hub.py` — an in-process `Hub` in mode `hub` on `127.0.0.1:0` (started as `cluster_kit` starts one, T3.1) with a tiny tic-tac-toe config (`learner.device: cpu`), cwd = a tmp project holding a copy of `examples/`; fake nodes from `hub_kit.FakeNode` (T2.3). If `FakeNode` cannot set `Hello` fields, add a keyword `hello_fields: dict | None` merged into its `Hello` (and `hardware_fields` into `Hello.hardware`), defaulting to values that pass (`code_hash` = the hub's, this interpreter's Python / torch, free bytes 1 TiB).
- `test_hello_refusals_reach_the_node` (parametrized `code_hash="0"*64`, `hardware.python="3.11.9"`, `hardware.torch_version="2.13.0"`, `hardware.os="Windows"`, `hardware.shm_free_bytes=1 << 20`): the session call ends with `grpc.StatusCode.FAILED_PRECONDITION`; `registration.parse_refusal(details)` has the expected reason; for `code_hash` its `code_hash` equals the hub's bundle hash; the hub's registry has no online node; `main.log` has one WARNING naming the node and the reason (a second attempt adds none).
- `test_source_node_on_loopback_is_accepted_without_code_hash`: `code_hash=""` from 127.0.0.1 → `Welcome` arrives.
- `test_torch_build_difference_warns_once`: `hardware.torch_build="rocm9.9"` → `Welcome`; reconnect twice; exactly one WARNING `torch build rocm9.9 on the node, the hub runs <build>` in `main.log`.
- `test_hub_prints_join_commands_and_serves_bootstrap`: a `colosseum hub -c <cfg> --listen 127.0.0.1 --port 0` process (`cli_runner.start_hub(..., cwd=project)`): stdout gets `Hub listening on 127.0.0.1, port <P> (TLS).`, `  colosseum join 127.0.0.1:<P> --token '<token>'` (the token of `HubIdentity.load_or_create()` under the test's `XDG_DATA_HOME`) and a curl line with `--pinnedpubkey 'sha256//<b64 of the fingerprint>'`; `logs/main.log` has `<secret: colosseum hub token>` and not the secret; the loader's `open_channel` + `fetch_info` over TLS return a code hash equal to `bundle.build_bundle(project, ..., out_dir=tmp)`'s for the same project and roots (deterministic hash); stop the hub with SIGINT (exit 130).
- `test_plain_train_serves_no_bootstrap`: a `train` hub without `--listen`; `Bootstrap/Info` on its loopback address (`hub.json`) with `insecure=True` → `UNIMPLEMENTED` (the loader's `LoaderError` mentions `colosseum train --listen`).
- `test_hub_logs_the_wsl_hint`: monkeypatch `colosseum.hub.joining.wsl_network` to return `WslNetwork("nat", "172.31.101.51", "wslinfo")`, start the in-process hub with `listen=None` (all interfaces) on port 0 → `main.log` has the netsh portproxy line with `connectaddress=172.31.101.51` at WARNING; with `listen="127.0.0.1"` it has none.

`tests/integration/test_sp5_join_node.py` — `NodeAgent` (in a thread, `exec_fn` recording) against `node_kit.FakeHub` (T2.4). Add to `FakeHub`, if missing: `refuse_session(code, details)` (abort the next sessions), `refuse_register(code, details)`, and access to the received `Hello`s.
- `test_hello_carries_code_hash_torch_and_free_space`: `NodeOptions(code_hash="c"*64, bundle_dir=tmp)` → the recorded `Hello.code_hash == "c"*64`; `hardware.torch_version, torch_build == torch_identity()`; `shm_free_bytes > 0`, `cache_free_bytes > 0`.
- `test_code_hash_refusal_reexecs_the_loader`: the FakeHub refuses with `Refusal("code_hash", "...", "e"*64).details()` → `exec_fn` called once with `sys.executable`, argv `[sys.executable, "-I", <colosseum/bootstrap/loader.py of this tree>, *join_argv(options)]`, env with `COLOSSEUM_TOKEN == str(options.token)`; no child is left running; the log has `code changed (cccccccccccc -> eeeeeeeeeeee)`.
- `test_code_hash_refusal_of_a_source_node_exits_1`: `bundle_dir=None` → `run()` returns 1, `exec_fn` not called, message names `colosseum join`.
- `test_other_refusals_exit_1_with_the_message` (parametrized `python`, `torch`, `shm`, `platform`): `run()` returns 1; the log has the refusal's message; no reconnect after it (the FakeHub sees one `Hello`).
- `test_unknown_node_registers_again_once`: the first session `UNAUTHENTICATED` → the agent calls `Register` again (FakeHub records a `RegisterRequest` with the stored node id), the second session gets `Welcome`; with `UNAUTHENTICATED` on every session → `run()` returns 1 after exactly two `Register` calls in total.
- `test_banned_node_exits_1`: `Register` → `PERMISSION_DENIED` → `run()` returns 1, message contains `banned`.
- `test_rotated_hub_key_stops_the_node`: the FakeHub serves TLS with another identity than the token's → `run()` returns 1, the message says the key does not match the token.

`tests/integration/test_sp5_join_cli.py` (CLI processes through `cli_runner`):
- `test_join_cli_argument_errors_exit_2`: `colosseum join h:1 --token T --token-file F`, `--workers -1`, `--gpus a,b`, no token anywhere (`COLOSSEUM_TOKEN` removed from the env) → exit 2 each; stderr names the problem.
- `test_join_cli_token_sources`: with a hub that is not running (a free port from a socket bound and closed by the test is acceptable here: nothing listens) and `COLOSSEUM_TOKEN` / `--token-file` / `--token` each → exit 1 with `cannot reach the hub` (the token was accepted, the connection failed).
- `test_join_help_lists_the_spec_flags`: `colosseum join --help` lists `--token`, `--token-file`, `--workers`, `--evaluators`, `--cpu-learners`, `--no-learners`, `--gpus`, `--name`, `--tags`, `--once`, `--insecure` (criterion 15: flags = `--help`).

`tests/integration/test_sp5_join_process.py` — the whole chain with real processes (2 loaded processes: the node's worker and learner):
- `test_join_process_trains_to_budget_once`: a tmp project with `examples/` copied from the repo and `configs/examples/tic_tac_toe.yaml` copied; token file from `HubIdentity.load_or_create()` (the test's `XDG_DATA_HOME`); `colosseum hub -c <cfg> --listen 127.0.0.1 --port 0` with `cli_runner.TINY` settings, `learner.device=cpu`, `evaluation.enabled=false`, run dir under `tmp_path`, cwd = the project; `address = cli_runner.hub_address(run_root)`; `colosseum join <address> --token-file F --name n1 --workers 1 --cpu-learners 1 --evaluators 0 --once` with `env = cli_runner.node_env(tmp_path, "n1")` (`COLOSSEUM_NODE_PYTHON` = this interpreter: no uv). Expected: the hub exits 0 (budget), the join process exits 0 (`--once`); the journal (`MatchLogReader`) has training matches, every one with `node == 1`; `tmp_path/n1/cache/colosseum/bundles/<hash>/project/examples/` exists and the node's log (under `node_log_dir`) names that bundle directory as its code; no process of either session is left. Expected duration about 15–20 s; if it takes more than 20 s on the owner's machine, mark it `slow` and say so in the report.

- [ ] **Step 5: (a) Run them (RED)**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/integration/test_sp5_join_hub.py tests/integration/test_sp5_join_node.py tests/integration/test_sp5_join_cli.py tests/integration/test_sp5_join_process.py -q`
Expected: failures (`colosseum join` is not a command; no `Hello` check; no `policy` keyword; `exec_fn` unknown to `NodeAgent`).

- [ ] **Step 6: (a) Hub side: policy, Hello check, bootstrap wiring, start-up messages**

- `hub/server.py`: `build_server(..., policy=None)`; in the Session servicer, after the first message: the `check_hello` call and the abort as described (no policy → no check); `SessionOpened` gains `warnings: tuple[str, ...] = ()`.
- `hub/hub.py`: `_build_bootstrap()` (modes `hub` / `idle`, `train` with `listen`), `_registration_policy()` (with `_weights_bytes()`), passing `bootstrap=` and `policy=lambda: self._policy` to `build_server`; `_print_startup_messages(port)` after the bind; `_on_session_opened` logs `warnings` once per (node id, text); `_on_register` refuses a banned `node_id` or `machine_id` with `PERMISSION_DENIED` and the ban hint (if T2.3 / T2.5 already do, keep theirs and the test).

- [ ] **Step 7: (a) Node side and the CLI**

- `node/session.py`: the classification of terminal statuses and the pin failure into `SessionRefused` / `SessionUnauthenticated` / `SessionBanned` / `SessionKeyMismatch` (reconnects stop after them).
- `node/agent.py`: `NodeAgent(options, *, exec_fn=os.execve)`; the `Hello` fields; the handlers of the four events; `_reexec_loader`; `main` as described; the start-up log lines.
- `cli.py`: the `join` command:
```python
@main.command("join")
@click.argument("address")
@click.option("--token", default=None, help="The hub's token (cst1...; `colosseum hub token` on the hub machine prints "
                                            "it). Also read from COLOSSEUM_TOKEN.")
@click.option("--token-file", default=None, type=click.Path(exists=True, dir_okay=False),
              help="Read the token from this file (keeps it out of the shell history).")
@click.option("--workers", type=click.IntRange(min=0), default=None,
              help="Worker processes [default: physical cores - 1].")
@click.option("--evaluators", type=click.IntRange(min=0), default=None,
              help="Evaluator processes (pulse, tournaments) [default: 1 with >= 4 physical cores, else 0].")
@click.option("--cpu-learners", type=click.IntRange(min=0), default=None,
              help="CPU learner slots [default: 1 with >= 4 physical cores, else 0].")
@click.option("--no-learners", is_flag=True, help="Run no learner on this machine.")
@click.option("--gpus", default=None, help="GPU indices for learner slots, e.g. 0,1 [default: every GPU].")
@click.option("--name", default=None, help="Node name [default: the host name; -2, -3 ... on a collision].")
@click.option("--tags", default=None, help="Comma-separated tags (learner.placement.tags).")
@click.option("--once", is_flag=True, help="Leave after the current run [default: stay for the hub's next runs].")
@click.option("--insecure", is_flag=True, help="Plain connection to a hub started with --insecure.")
def join_cmd(address: str, token: str | None, token_file: str | None, workers: int | None, evaluators: int | None,
             cpu_learners: int | None, no_learners: bool, gpus: str | None, name: str | None, tags: str | None,
             once: bool, insecure: bool) -> None:
    """Join this machine to a hub: fetch the hub's code, build its environment with uv, run a node.

    Exit code: 0 normal stop (--once after the run, nodes remove), 1 refused or failed, 2 bad arguments,
    130 SIGINT, 143 SIGTERM.
    """
    import re

    if token and token_file:
        raise click.UsageError("--token and --token-file are mutually exclusive")
    if gpus is not None and not re.fullmatch(r"\d+(,\d+)*", gpus):
        raise click.BadParameter("a comma-separated list of GPU indices, e.g. 0,1", param_hint="--gpus")
    from colosseum.bootstrap.loader import loader_main

    argv = [address]
    for flag, value in (("--token", token), ("--token-file", token_file), ("--workers", workers),
                        ("--evaluators", evaluators), ("--cpu-learners", cpu_learners), ("--gpus", gpus),
                        ("--name", name), ("--tags", tags)):
        if value is not None:
            argv += [flag, str(value)]
    argv += [flag for flag, on in (("--no-learners", no_learners), ("--once", once), ("--insecure", insecure)) if on]
    sys.exit(loader_main(argv))
```
- `tests/cli_runner.py`: `node_env`, `join_cmd`, `start_join`, `hub_address` (reads `hub.json` through `colosseum.hub.registry.read_hub_json`, waiting by file presence), `start_hub(..., cwd=)`.

- [ ] **Step 8: (a) Run the (a) tests (GREEN)**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_join_checks.py tests/unit/test_sp5_join_hints.py tests/unit/test_sp5_join_slots.py tests/integration/test_sp5_join_hub.py tests/integration/test_sp5_join_node.py tests/integration/test_sp5_join_cli.py tests/integration/test_sp5_join_process.py -q -rw`
Expected: all pass. Also run the cluster tests of T3.x–T5.x (`tests/integration/test_sp5_*cluster*`, or the files that import `cluster_kit`): their in-process nodes are source nodes and must still be accepted.

- [ ] **Step 9: (a) Full fast suite + ruff, commit, push**

Run the full fast suite and ruff (green, zero warnings), then:
```bash
git add src/colosseum/net/registration.py src/colosseum/hub/joining.py src/colosseum/hub/server.py \
        src/colosseum/hub/hub.py src/colosseum/node/agent.py src/colosseum/node/session.py src/colosseum/cli.py \
        tests/cli_runner.py tests/hub_kit.py tests/node_kit.py tests/unit/test_sp5_join_checks.py \
        tests/unit/test_sp5_join_hints.py tests/unit/test_sp5_join_slots.py tests/integration/test_sp5_join_hub.py \
        tests/integration/test_sp5_join_node.py tests/integration/test_sp5_join_cli.py \
        tests/integration/test_sp5_join_process.py
git commit -m "feat: colosseum join, registration checks and the node's self-restart on new code"
git push
```

- [ ] **Step 10: (b) Write the failing tests of persistent nodes and node admin**

`tests/integration/test_sp5_join_node.py` (extend; FakeHub):
- `test_release_stops_the_agent_with_exit_0`: `Command(kind="release")` → the node's stop sequence runs (a learner child sends its final snapshot `Upload` before the session closes: FakeHub records it), `run()` returns 0 without `--once`.
- `test_once_exits_after_stop_run`: `Command(kind="stop_run")` with `once=True` → `run()` returns 0 after the stop sequence.
- `test_without_once_waits_for_the_next_run`: the same without `once` → the agent keeps running; the FakeHub closes the stream and accepts a new session: a second `Hello` arrives (with the same node id); `request_stop()` ends it (exit 130 path of T4.4 or 0, as T4.4 defines for `request_stop`).
- `test_new_run_id_switches_run`: `Welcome(run_id="A")`, a result batch not acked; the FakeHub closes the stream; the next `Welcome(run_id="B")` → one WARNING `1 result batches of run A were never acknowledged (the run ended): dropped`; the next `LineupRequest` cells carry `epoch 0` (no live cells of run A); the node's log file now lives under `node_log_dir(ca8, "B")`; no file of the run-A cache prefix is left in the shm dir.

`tests/integration/test_sp5_nodes_admin.py` (in-process hub, `FakeNode`s, the admin client of T2.5; CLI processes where named):
- `test_nodes_list_shows_online_and_offline_nodes`: two fake nodes register, one keeps its session → `admin.call(target, "nodes.list")["data"]["nodes"]` has both rows ordered by number, states `online` / `offline`, the online row's hardware fields from its `Hello`; `colosseum nodes list --hub <addr> --token-file F` (process) prints a table whose rows contain both names and a header line starting with `#`.
- `test_format_nodes_table`: rows with and without hardware, `banned: 2` as the last line when `banned` is 2.
- `test_nodes_remove_releases_and_forgets`: the online fake node receives `Command(kind="release")`; it then closes its session → the reply has `code 0` and `released`; the `NodeBook` record is gone; a new session with the old secret → `UNAUTHENTICATED`; a new `Register` with the token → accepted (a new number: numbers are never reused).
- `test_nodes_remove_ban_refuses_registration`: `{"name": ..., "ban": True}` → after the session closes, `Register` with the same `machine_id` (any node id) → `PERMISSION_DENIED`; `colosseum nodes list` shows `banned: 1`.
- `test_nodes_remove_deadline`: with `distributed.shutdown_grace_sec: 1` and a fake node that ignores `release` → the reply arrives after the hub's grace + 5 s deadline with `did not stop`, and the record is removed anyway.
- `test_nodes_remove_unknown_name_exit_1`: `colosseum nodes remove ghost --hub <addr> --token-file F` → exit 1, stderr `no node named ghost`.
- `test_hub_token_prints_the_machine_token`: `colosseum hub token` (process, the test's `XDG_DATA_HOME`) prints one line equal to `str(HubIdentity.load_or_create().token)`; a second call prints the same.
- `test_hub_token_rotate_revokes_and_refuses_while_running`: `registry.register_running(identity_dir, HubInfo(pid=os.getpid(), ...))` → `colosseum hub token --rotate` exits 1 with `stop it first`; after `unregister_running` → exit 0, a new token (other secret, other fingerprint), `nodes.json` without nodes, stderr `every node is revoked`.

`tests/integration/test_sp5_join_persistent_slow.py` (`@pytest.mark.slow`; 2 loaded processes):
- `test_node_takes_the_next_run_and_new_code`: hub run 1 as in `test_join_process_trains_to_budget_once` but the join process without `--once`; hub 1 exits 0; the node keeps running (its log has `waiting for the next run`); append a comment line to `examples/tic_tac_toe/__init__.py` of the project (a new code hash); hub run 2 with another run name on the same port (`--port <P of run 1>`: the port the node knows); the node is refused `code_hash`, restarts through the loader (its stderr / log: `downloading the code bundle <new hash12>`, `code changed`), joins run 2 with the same node number and name (journal `node == 1` in both runs; `nodes list` of run 2 shows `n1`), and run 2 reaches its budget (exit 0); then SIGINT to the join process group → it exits within 10 s (code 130). The old bundle stays in the cache (newest 5 kept).

`tests/integration/test_sp5_join_one_liner_slow.py` (`@pytest.mark.slow`; 2 loaded processes; skipped without a curl that has HTTP/2):
- `test_one_liner_joins_a_real_hub`: hub run as above; take the curl line for `127.0.0.1:<P>` from the hub's stdout verbatim and run `sh -c "<line> -s -- --name n1 --workers 1 --cpu-learners 1 --evaluators 0 --once"` (the documented way to pass join flags) with `env = node_env(...)`, `COLOSSEUM_UV` = `join_kit.write_fakes(...)`'s fake uv (its `uv run` executes this interpreter: the real loader runs), `COLOSSEUM_NODE_PYTHON` = this interpreter → the hub reaches its budget, the `sh` process exits 0, the journal has matches of node 1, the installer's loader copy exists under `tmp_path/n1/cache/colosseum/loader/`. This is criterion 8's mechanism on one machine without network installs.

- [ ] **Step 11: (b) Run them (RED)**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/integration/test_sp5_join_node.py tests/integration/test_sp5_nodes_admin.py -q` and `env -u PYTHONPATH .venv/bin/python -m pytest -m slow tests/integration/test_sp5_join_persistent_slow.py tests/integration/test_sp5_join_one_liner_slow.py -v`
Expected: failures (no `release` / `stop_run` exit rules, no `_switch_run`, unknown admin commands, no `nodes` / `hub token` commands).

- [ ] **Step 12: (b) Persistent nodes in the agent**

`node/agent.py`: `_switch_run(welcome)`; the `stop_run` and `release` rules (exit 0 with `--once` / after `release`; stay otherwise, INFO `run <id> finished; waiting for the next run of this hub (Ctrl+C to leave)`).

- [ ] **Step 13: (b) Node admin in the hub**

`hub/hub.py`: `_node_rows()`, `_on_admin` for `nodes.list` and `nodes.remove`, the pending-removal table (`node_id → (ban, reply, deadline)`) completed in `_on_session_closed` or by the main loop's deadline check; the immediate placement / cell handling through the node-loss path with reason `removed`. `hub/admin.py`: `format_nodes_table(rows)`.

- [ ] **Step 14: (b) The CLI**

`cli.py`:
- `@main.group("nodes")` "Machines joined to a running hub." with `list` and `remove NAME [--ban]`, both with `--hub ADDR`, `--run RUN`, `--token`, `--token-file` → `admin.resolve_hub(...)` → `admin.call(target, "nodes.list" | "nodes.remove", args)`; a reply `code != 0` or a `SecurityError` → its text on stderr, exit 1; `nodes list` prints `format_nodes_table`.
- `hub token [--rotate]` as a subcommand of T2.5's `hub` group, as described.

- [ ] **Step 15: (b) Run the (b) tests (GREEN)**

Run the commands of Step 11. Expected: all pass; report the two slow tests' durations (expected 40–70 s and 25–40 s).

- [ ] **Step 16: (b) Full fast suite + ruff**

Run the full fast suite and ruff. Expected: green, zero warnings. (The task changes no training math, worker or learner: the slow suite runs only the two new slow tests above.)

- [ ] **Step 17: (b) Commit and push**

```bash
git add src/colosseum/node/agent.py src/colosseum/hub/hub.py src/colosseum/hub/admin.py src/colosseum/cli.py \
        tests/integration/test_sp5_join_node.py tests/integration/test_sp5_nodes_admin.py \
        tests/integration/test_sp5_join_persistent_slow.py tests/integration/test_sp5_join_one_liner_slow.py \
        tests/node_kit.py tests/hub_kit.py
git commit -m "feat: persistent nodes, nodes list / remove --ban, hub token --rotate"
git push
```

---

## Contract notes

### Additions and deviations versus `00-overview.md`

1. **`Bundle` gains `python_path` and `skipped`; `ImportRoot`, `import_roots`, `bundles_dir`, `bundle_python_path`, `project_dir`, `sha256_file`, `BundleError` and the constants are added; `build_bundle(..., roots=None)`** injects the import roots (tests; default `import_roots()`). Layout `bundle.json` + `project/` + `pkg/<i>/` (the overview's "tar of colosseum/ ..." made precise): an import root inside the project (a source checkout's `src/`) is only a `PYTHONPATH` entry, so the package is not stored twice.
2. **Exclusions beyond spec 2.3:** `wandb/` at the top (WandB writes its run files into the hub's cwd, which would change the code hash every run), `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `*.pyc`, `*.pyo`; refusal of a home / root project dir and of more than 50 000 files or 1 GiB (safety: a hub started from the wrong directory). The hub adds its `run.dir` when it is inside the project.
3. **`requirements.EXCLUDED` also lists `cuda-*` and `pytorch-triton*`** (the overview has `torch`, `nvidia-*`, `triton`, `colosseum`): torch 2.14.1's CUDA wheels pull `cuda-bindings`, `cuda-pathfinder`, `cuda-toolkit` (seen in this `.venv`), which a CPU or macOS node must not pin. Distributions from local paths (`direct_url.json` `file:`) are excluded too (spec 2.4). `read_requirements_file`, `hub_freeze`, `canonical_name`, `excluded`, `requirements_hash` are added.
4. **`torch_spec(hub_torch, hub_build, *, platform: NodePlatform, gpus: Sequence[NvidiaGpu]) -> TorchChoice(requirement, backend, cuda_driver, warning)`** instead of `torch_spec(hub_torch, hub_build, probe: Hardware) -> (requirement, index url, warning)`: T6.1 is in group A, before T2.4 creates `Hardware`, and the loader must run without colosseum; uv's `--torch-backend` (plus `UV_CUDA_DRIVER_VERSION`) replaces an index URL (uv then also takes torch's `nvidia-*` / `triton` set right). `TorchSelectError`, `probe_nvidia`, `probe_platform`, `nvidia_smi_environ`, `torch_identity`, `build_tag`, `parse_version`, `public_version`, `cuda_major` are added.
5. **Environment directory `envs/<key>/` with `key = sha256(requirements_hash|python|torch_version|backend)[:24]`** instead of `envs/<requirements hash>/`: the same requirement list needs different environments for different Python versions and torch builds.
6. **`loader_main(argv, *, environ=None, exec_fn=None)`** (test hooks; the contract's call `loader_main(argv)` works unchanged). The loader is **self-contained** (stdlib + grpcio, no protobuf, no colosseum): its own wire codec for the three `Bootstrap` messages and its own SPKI parse; it loads `bundle.py` / `torch_select.py` as siblings. Its public helpers are listed in T6.2's Interfaces. Environment variables `COLOSSEUM_NODE_PYTHON` (use this interpreter instead of building an environment: tests, pre-built images) and `COLOSSEUM_UV` (the uv binary) are added; the hidden loader flag `--uv` passes the installer's uv.
7. **`installer_script(*, insecure, python, grpcio, uv_version=UV_VERSION)`** instead of `installer_script(*, hub_address, fingerprint, insecure)`, and **`one_liner(address, token, *, insecure)`** instead of `one_liner(address, fingerprint)`: one script serves every address of the hub and the SSH mode (T8.1 runs it with `sh -s` through `ssh`), so the address comes from `COLOSSEUM_HUB`, which the one-liner sets next to `COLOSSEUM_TOKEN` (`... | tail -c +6 | COLOSSEUM_HUB='<addr>' COLOSSEUM_TOKEN='<token>' sh`: the spec's command plus one variable); the fingerprint is in the token; the one-liner needs the token anyway. For `--insecure` hubs the one-liner uses `--http2-prior-knowledge` and `http://`. `join_command`, `curl_pin`, `END_MARKER`, `SCRIPT_PATH` are added.
8. **`BootstrapService`** (referenced by the overview's `build_server(..., bootstrap: BootstrapService | None)` but not defined) is `colosseum.bootstrap.installer.BootstrapService(bundle, requirements, requirements_hash, python, torch_version, torch_build, script, protocol, archive)` with `.info()`; built by `make_bootstrap_service(...)`; the archive is served from memory, hub archives live in `<cache_home>/colosseum/hub-bundles/` (newest 5 kept).
9. **`colosseum.hub.server.add_bootstrap(server, service)` and `BootstrapServicer(service)`** are the T6.2 additions to `hub/server.py`; `Bootstrap/Bundle` answers `NOT_FOUND` for another code hash.
10. **New module `colosseum.net.registration`** (T6.3; not in the overview's file map): `RegistrationPolicy`, `Refusal`, `parse_refusal`, `check_hello`, `need_bytes`, `is_loopback_peer` — shared by the hub's Session servicer and the node (which reads a refusal from the gRPC status). **New module `colosseum.hub.joining`** (T6.3): addresses, join instructions, WSL2 detection and hint, `startup_messages`.
11. **Refusals travel as gRPC status `FAILED_PRECONDITION` with JSON details** `{"refused": <reason>, "message": str, "code_hash": str}` before `Welcome` (the proto has no refusal message; reasons `platform`, `protocol`, `code_hash`, `python`, `torch`, `shm`, `disk`). The check runs in the Session servicer thread against an immutable policy (`build_server(..., policy=)`, addition), so a refused node never reaches the main loop; `SessionOpened` gains `warnings`.
12. **`NodeOptions` gains `code_hash: str = ""` and `bundle_dir: Path | None = None`; `NodeAgent(options, *, exec_fn=os.execve)`;** `node.agent` gains `join_slots`, `build_agent_parser`, `options_from_args`, `join_argv`, `_reexec_loader`, `_switch_run`; `node.session` gains the events `SessionRefused`, `SessionUnauthenticated`, `SessionBanned`, `SessionKeyMismatch`. The agent's command line is the loader's arguments plus `--code-hash H --bundle DIR`, the token only in `COLOSSEUM_TOKEN`, popped from the agent's environment after reading.
13. **`Command(kind="release")`** is added to the command kinds (the overview lists `release`; its meaning is fixed here: the node stops as at the end of a run and exits 0); admin commands `nodes.list` / `nodes.remove` get the argument and reply shapes of T6.3; `admin.format_nodes_table` is added.
14. **`tests/cli_runner.py`** gains `node_env`, `join_cmd`, `start_join`, `hub_address`, and `start_hub(..., cwd=)`; `tests/join_kit.py` (T6.2) is new; `hub_kit.FakeNode` / `node_kit.FakeHub` may gain the knobs T6.3 lists (`hello_fields`, `refuse_session`, `refuse_register`).

### Proposed rulings (what — why — cost if wrong)

1. **The loader is a self-contained file with its own codec for the three `Bootstrap` messages** (no protobuf, no generated module, no colosseum import) — why: the one-liner's machine has no colosseum, and `colosseum join` may run an installed colosseum of another version; `Bootstrap` becomes the one wire contract that must stay compatible (fields added, never renumbered); a test compares the loader's field numbers with the generated descriptors — cost if wrong: a renumbered `Bootstrap` field breaks old loaders (the test catches it in this repo; old installed versions would need the one-liner).
2. **An empty `Hello.code_hash` is accepted only from a loopback peer** ("source node": `train`'s local node, in-process test nodes) — why: those run the hub's own source tree and have no bundle, and the T3.x–T5.x cluster tests start such nodes; a remote node always comes through the loader with a hash — cost if wrong: a stale source checkout started by hand on the hub machine is not caught by the code-hash check (it still meets the protocol, Python and torch checks).
3. **Torch build rule for old GPUs:** compute capability < 7.5 on any GPU → `cu126` while the hub's torch is ≤ 2.14 and the driver supports CUDA 12, else the CPU build with a WARNING; otherwise the hub's CUDA build when the driver supports it, else uv's `auto` with the probed driver — why: uv's `auto` ignores the compute capability (uv #14742) and CUDA 13 wheels have no kernels below sm_75; PyTorch 2.14 is the last release with CUDA 12 wheels — cost if wrong: an old-GPU node trains its learner on the CPU after 2.14 (stated in the WARNING).
4. **`nodes remove` = the run-stop sequence of that node** (learners' final snapshots, results flushed, `shutdown_grace_sec`), not "play every current episode to its end"; the hub changes `nodes.json` / the ban list only after the node's session closed (or after grace + 5 s) — why: the worker has no "finish the episode" mode and an episode can last arbitrarily long; the final snapshot `Upload` must still authenticate — cost if wrong: the last seconds of the node's play are lost (as on a node loss; no PFSP bias: unfinished matches are not reported).
5. **The token is printed only to the hub's stdout, never to `main.log`** (the log gets a redacted copy) — why: run dirs are shared and published as reports; the token is full admin access — cost if wrong: a user who looks for the join command in `main.log` finds `<secret: colosseum hub token>` and must run `colosseum hub token`.
6. **`hub token --rotate` and banned machines:** the part does not decide whether `rotate()` empties `banned.json` (T1.3 owns it); after a rotation every machine needs the new token anyway — cost if wrong: none for security.
7. **WSL2 hint whenever the hub listens beyond loopback inside WSL2 and the mode is not `mirrored` / `bridged`** (`nat`, `consomme`, `none`, `unknown`), with netsh portproxy first — why: `wslinfo` reports `nat` even when the route goes through Consomme (`wsltap`) on the owner's machine, and Windows 10 (the owner's) has no mirrored mode — cost if wrong: a mirrored-mode user with an old WSL (no `wslinfo`) and a 172.16/12 address sees a needless hint.
8. **Private uv and Python live in the colosseum cache** (`<cache>/colosseum/uv/`, `uv-cache/`, `python/`) unless the user set `UV_CACHE_DIR` / `UV_PYTHON_INSTALL_DIR` — why: same filesystem as the environments (hard links), the managed Python must outlive the environments, no edits of the user's profile or uv receipt — cost if wrong: one extra torch download on machines that already had it in uv's default cache.
9. **Pruning:** a node keeps the newest 5 bundles and 2 environments; the hub keeps 5 archives — why: an environment with CUDA torch is several GB — cost if wrong: switching between three hubs with different requirement lists rebuilds environments.

### Assumptions about other parts

1. **T1.2** numbers `BootstrapInfo`, `BundleRequest`, `BlobPart` in the overview's declaration order (1, 2, ...); `Bootstrap/Info` takes an empty message; `Hello.code_hash`, `Hello.protocol_major` and `Hardware.{os, python, torch_version, torch_build, shm_free_bytes, cache_free_bytes}` exist as in the outline.
2. **T1.3:** `HubIdentity.load_or_create(path)` with `.token` (`.secret`, `.ca_fingerprint`, `str()` = the token), `.fingerprint`, `.ca_pem` (the CA = the TLS server certificate, R15), `.nodes`, `.server_credentials()`, `.rotate()`; `spki_fingerprint(pem)`; `AuthInterceptor` leaves `Bootstrap/Info` and `Bootstrap/Script` open and checks `x-colosseum-token` (the secret) for `Bootstrap/Bundle`; `NodeBook.register(node_id=<known id>)` keeps the record's number and name (new secret), an unknown id gets a new number; `NodeBook.is_banned(node_id, machine_id)`; `parse_token`, `TOKEN_ENV`, `SecurityError`; `server_options(max_message_mb)`.
3. **T2.3:** `send_parts(data: bytes | Path)` yields parts with `offset` (bytes before the part), `total_size` and `last`; `build_server(..., bootstrap=)` exists; the Session servicer reads `Hello` as the first message before posting `SessionOpened`; `HubSession` reports a terminated stream's status to its owner (T6.3 adds the classification); `hub_kit.FakeNode` can register and open sessions.
4. **T2.4:** `NodeOptions` as in the overview (frozen dataclass); `Hardware` with `gpus` as dicts that carry `index`; `default_slots(hw)` returns the `Slots` fields as a dict (`workers`, `evaluators`, `cpu_learners`, `gpus` list, `no_learners`); `node.agent.main(argv)` exists (T6.3 replaces its parsing); `hardware.probe()` honours cgroup quotas and affinity (its tests are T2.4's); `node_kit.FakeHub` serves Register / Session in process. If T2.4 computes `Hardware.torch_version / torch_build` itself, it should use `torch_select.torch_identity()` (T6.1) so the hub and the node compare the same strings.
5. **T2.5:** `colosseum hub` accepts `--port 0` and records the bound address in `runs/<name>/hub.json`; the `hub` click group runs a hub only without a subcommand; `admin.call` returns `{"code", "text", "data"}`; `cli_runner.start_hub` exists; `running_hubs` / `register_running` / `HubInfo` as in the contract.
6. **T3.2:** a `join` node opens its data port from the hub's `distributed.data_port` as T3.2 implements for non-local nodes; `node.agent.main` does not set `NodeOptions.data_port` beyond what T3.2 expects of a join node.
7. **T4.2 / T4.4:** a node-loss path that drops a node's cells and relocates its learners can be called with a reason and without the `learner_relocate_after_sec` wait (`nodes remove`); the node's stop sequence (final snapshots by the dedicated upload, result flush, `shutdown_grace_sec`) is callable for `stop_run` and `release`; the hub-away and `--once` + hub-lost behaviour is T4.4's.
8. **T4.3:** the journal's `node` column exists (the join process tests read it).
9. **T7.1** builds `nodes.json` from `Hub._node_rows()`; **T7.3**'s node image runs `colosseum join` (the loader builds the environment into the cache volume); **T7.5** writes `docs/DISTRIBUTED_GUIDE.md` with a section «WSL2» (netsh portproxy + `New-NetFirewallRule` first: the owner's Windows 10 19045 has no mirrored mode; then mirrored + Hyper-V rule for Windows 11 22H2+; Tailscale), how to pass join flags to the one-liner (`... sh -s -- --workers 4`), `COLOSSEUM_NODE_PYTHON`, and the cache layout and pruning; **T7.6**'s runbook runs the one-liner on the laptop (criterion 8: the first real environment build, with network) and checks the WARNING-free torch build line on the PC's GPU node.
10. **T8.1** feeds `installer_script` to `ssh <host> 'COLOSSEUM_HUB=127.0.0.1:<tunnel port> COLOSSEUM_TOKEN=... sh -s -- <flags>'`; **T8.3** sends `Command(kind="update_code")` and the node calls `_reexec_loader()`.
