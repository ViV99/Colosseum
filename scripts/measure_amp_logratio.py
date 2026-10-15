"""Zero-lag log-ratio of the CUDA learner under AMP (docs/GPU_CHECKS.md, final SP2 review, area A, M-6).

Under AMP ``float16`` / ``bfloat16`` the Gaussian log-prob and entropy (also for ``Box`` inside ``Units``)
run in low precision: the recorded float32 actions are cast to ``mean.dtype``. The script measures what
that does to the importance ratio at zero policy lag:

- roles: observation ``Box(-1, 1, (16,))``; actions ``Units(6, Box(-1, 1, (4,)))`` (``units_box4``),
  ``Units(6, Dict(move=Discrete(5), aim=Box(-1, 1, (2,))))`` (``units_dict``) and ``Units(6, Discrete(5))``
  (``units_discrete``, the baseline: categorical logits are fp16 / bf16 under autocast too);
- models ``make_test_model(role, core, hidden)`` for (core, hidden) in (none, 64), (lstm, 256); every
  ``log_std`` parameter filled with 0, -2 and -4 (roles with a ``Box`` component; only 0 for
  ``units_discrete``);
- 8 chunks from ``synthetic_chunk(..., logp_noise=0.0, random_units=True)``: the behaviour log-probs come
  from the CPU fp32 model, as on a worker, so the learner's weights equal the behaviour weights;
- for fp32, AMP fp16 and AMP bf16, ``APPO(..., device="cuda")`` evaluates the chunks on the training path
  (``_evaluate(_prepare_batch(chunks))`` under ``torch.no_grad()``). Printed: mean / p95 / max of the
  per-decider |log-ratio| over valid ``act`` deciders, of the joint |log-ratio| over ``act`` slots, the
  per-decider difference to the fp32 learner, and ``compute_loss``'s ``log_rho_abs_p95``,
  ``clip_fraction``, ``rho_clip_frac``.

At zero lag every log-ratio would be 0 in exact arithmetic; the spread is pure numerics. The fp32 column
of the LSTM core includes cuDNN TF32 (torch's default for cuDNN RNN on Ampere+, ruling G-R1); the fp32
column of the ``none`` core is plain fp32 noise.

Flags (candidate fixes, measured, not applied in the framework):
- ``--fp32-head``: the policy head (``game_helpers.TreePolicyHead``) runs in float32 under autocast
  (autocast off inside the head, float inputs upcast);
- ``--fp32-gauss``: ``colosseum.networks.dist.units.gaussian_log_prob`` upcasts mean, log_std and the
  actions to float32.

Usage (needs CUDA; imports the test kit, so it puts ``tests/`` on ``sys.path``)::

    .venv/bin/python scripts/measure_amp_logratio.py [--fp32-head] [--fp32-gauss] [--json out.json]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gymnasium  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

OBS = gymnasium.spaces.Box(-1.0, 1.0, (16,), np.float32)
MODES = {
    "fp32": {"use_amp": False},
    "fp16": {"use_amp": True, "amp_dtype": "float16"},
    "bf16": {"use_amp": True, "amp_dtype": "bfloat16"},
}
CORES = (("none", 64), ("lstm", 256))
LOG_STDS = (0.0, -2.0, -4.0)
NUM_CHUNKS = 8
PATTERN = "A" * 63 + "B"
DIAGNOSTICS = ("log_rho_abs_p95", "clip_fraction", "rho_clip_frac")


def _roles() -> dict:
    from colosseum.envs.game import RoleSpec
    from colosseum.envs.spaces import Units

    box = gymnasium.spaces.Box
    return {
        "units_box4": RoleSpec(OBS, Units(6, box(-1.0, 1.0, (4,), np.float32))),
        "units_dict": RoleSpec(OBS, Units(6, gymnasium.spaces.Dict({
            "move": gymnasium.spaces.Discrete(5), "aim": box(-1.0, 1.0, (2,), np.float32)}))),
        "units_discrete": RoleSpec(OBS, Units(6, gymnasium.spaces.Discrete(5))),
    }


def patch_fp32_head() -> None:
    """Run the test kit's policy head in float32 under autocast (candidate fix for M-6)."""
    import game_helpers

    orig = game_helpers.TreePolicyHead.forward

    def forward32(self, features, aux):
        with torch.autocast("cuda", enabled=False):
            aux32 = {k: (v.float() if v.is_floating_point() else v) for k, v in aux.items()}
            return orig(self, features.float(), aux32)

    game_helpers.TreePolicyHead.forward = forward32


def patch_fp32_gauss() -> None:
    """Compute the Units Gaussian log-prob in float32 (mean, log_std and actions upcast)."""
    import colosseum.networks.dist.units as units_mod

    orig = units_mod.gaussian_log_prob

    def gauss32(mean, log_std, x):
        return orig(mean.float(), log_std.float(), x.float())

    units_mod.gaussian_log_prob = gauss32


def _stats(x: torch.Tensor) -> dict:
    a = x.abs()
    return {"mean": a.mean().item(), "p95": a.quantile(0.95).item(), "max": a.max().item()}


def _fmt(s: dict) -> str:
    return f"mean {s['mean']:.2e}  p95 {s['p95']:.2e}  max {s['max']:.2e}"


def measure(role_name: str, role, log_std: float, core: str, hidden: int) -> list[dict]:
    from colosseum.algorithms.appo import APPO
    from colosseum.core.config import AlgorithmConfig
    from colosseum.core.specs import ActionSpec
    from colosseum.core.types import SLOT_ACT
    from game_helpers import make_test_model, synthetic_chunk

    torch.manual_seed(0)
    model = make_test_model(role, core=core, hidden=hidden)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "log_std" in name:
                p.fill_(log_std)
    chunks = [synthetic_chunk(model, role, PATTERN, seed=i, logp_noise=0.0, random_units=True)
              for i in range(NUM_CHUNKS)]
    spec = ActionSpec.from_space(role.action_space)
    beh_unit = torch.stack([c.behavior_unit_logp for c in chunks], dim=1)  # [S, B, K]
    beh_joint = torch.stack([c.behavior_logp for c in chunks], dim=1)      # [S, B]
    is_act = torch.stack([c.kind for c in chunks], dim=1) == SLOT_ACT      # [S, B]

    print(f"\n== {role_name}  core={core}  hidden={hidden}  log_std={log_std}")
    rows, ref = [], None
    for mode, kwargs in MODES.items():
        algo = APPO(copy.deepcopy(model), AlgorithmConfig(**kwargs), spec, device="cuda")
        with torch.no_grad():
            _, ev = algo._evaluate(algo._prepare_batch(chunks))
            metrics = algo.compute_loss(chunks)
        unit = ev["unit_log_probs"].cpu()
        valid = ev["unit_valid"].cpu() & is_act.unsqueeze(-1)
        row = {
            "role": role_name, "core": core, "hidden": hidden, "log_std": log_std, "mode": mode,
            "unit": _stats((unit - beh_unit)[valid]),
            "joint": _stats((ev["log_probs"].cpu() - beh_joint)[is_act]),
            "vs_fp32": None if ref is None else _stats((unit - ref)[valid]),
            "diagnostics": {k: float(metrics[k]) for k in DIAGNOSTICS},
        }
        ref = unit if ref is None else ref
        line = f"  {mode}: unit log-ratio {_fmt(row['unit'])} | joint {_fmt(row['joint'])}"
        if row["vs_fp32"] is not None:
            line += f" | vs fp32 learner {_fmt(row['vs_fp32'])}"
        print(line)
        print(f"        diagnostics: {row['diagnostics']}")
        rows.append(row)
    return rows


def _summary(rows: list[dict]) -> str:
    """Markdown table: per-decider |log-ratio| p95 / max per mode, bf16 clip_fraction."""
    by_key: dict[tuple, dict] = {}
    for r in rows:
        by_key.setdefault((r["role"], r["core"], r["log_std"]), {})[r["mode"]] = r
    lines = ["| role | core | log_std | fp32 p95 / max | fp16 p95 / max | bf16 p95 / max | bf16 clip_fraction |",
             "|---|---|---|---|---|---|---|"]
    for (role, core, log_std), m in by_key.items():
        cells = [f"{m[k]['unit']['p95']:.2e} / {m[k]['unit']['max']:.2e}" for k in MODES]
        clip = m["bf16"]["diagnostics"]["clip_fraction"]
        lines.append(f"| {role} | {core} | {log_std} | " + " | ".join(cells) + f" | {clip:.4g} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fp32-head", action="store_true", help="run the policy head in float32 under autocast")
    ap.add_argument("--fp32-gauss", action="store_true", help="upcast inside the Units gaussian_log_prob")
    ap.add_argument("--json", type=Path, default=None, help="write every measured number to this file")
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("measure_amp_logratio.py needs CUDA: torch.cuda.is_available() is False", file=sys.stderr)
        return 2
    if args.fp32_head:
        patch_fp32_head()
        print("policy head forced to float32 under autocast")
    if args.fp32_gauss:
        patch_fp32_gauss()
        print("Units Gaussian log-prob forced to float32")

    rows: list[dict] = []
    for role_name, role in _roles().items():
        log_stds = LOG_STDS if role_name != "units_discrete" else (0.0,)
        for log_std in log_stds:
            for core, hidden in CORES:
                rows.extend(measure(role_name, role, log_std, core, hidden))
    print("\nPer-decider |log-ratio| at zero lag (learner on CUDA vs CPU fp32 behaviour):\n")
    print(_summary(rows))
    if args.json is not None:
        payload = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                   "fp32_head": args.fp32_head, "fp32_gauss": args.fp32_gauss, "rows": rows}
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
