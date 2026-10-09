"""Generator of the SP2 checkpoint fixture (SP3 task T0.1). Run ONCE on SP2 code; the output is committed.

    .venv/bin/python tests/fixtures/sp2/make_sp2_checkpoint.py

Writes ``tests/fixtures/sp2/run/checkpoints/agent_0/ckpt_v3/`` (``model.pt``, ``trainer_state.pt``,
``meta.json``) for ``sp2_ttt_tiny.yaml``: three APPO updates on synthetic chunks, then the payload and
``meta.json`` exactly as SP2's launcher saves them (``learner.make_checkpoint_payload`` +
``Coordinator.save_checkpoint_payload`` + ``Launcher._save_checkpoint``'s meta). It uses only APIs that
existed in SP2 and is a record of how the fixture was made; SP3 tasks never re-run it.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "tests")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402

from colosseum.algorithms.appo import APPO  # noqa: E402
from colosseum.coordinator.checkpoint_manager import CheckpointManager  # noqa: E402
from colosseum.core.config import config_hash, load_config  # noqa: E402
from colosseum.core.registry import build_model, env_spec  # noqa: E402
from colosseum.core.roles import agent_role_spec, resolve_agent_roles, role_signature  # noqa: E402
from colosseum.core.specs import ActionSpec  # noqa: E402
from colosseum.learner.learner import make_checkpoint_payload  # noqa: E402
from game_helpers import synthetic_chunk  # noqa: E402

AGENT = "agent_0"
ENV_STEPS = 1536


def main() -> None:
    torch.manual_seed(0)
    config = load_config(HERE / "sp2_ttt_tiny.yaml")
    spec = env_spec(config)
    roles = resolve_agent_roles(config, spec)[AGENT]
    role = agent_role_spec(spec, roles)
    agent_config = config.get_agent_config(AGENT)
    model = build_model(agent_config, role)
    algo = APPO(model, agent_config.algorithm, ActionSpec.from_space(role.action_space), device="cpu")
    for step in range(3):
        chunks = [synthetic_chunk(model, role, "AAATAAAB", seed=10 * step + i, agent_id=AGENT) for i in range(2)]
        algo.train_step(chunks)
    payload = make_checkpoint_payload(AGENT, algo, final=True)
    shutil.rmtree(HERE / "run", ignore_errors=True)
    meta = {
        "final": bool(payload["final"]),
        "networks": agent_config.networks.model_dump(mode="json", by_alias=True),
        "config_hash": config_hash(config),
        "env_steps": ENV_STEPS,
        "roles": list(roles),
        "role_signature": role_signature(role),
    }
    out = HERE / "run" / "checkpoints"
    ckpt_id = CheckpointManager(out).save(
        agent_id=AGENT, policy_version=int(payload["policy_version"]), model_state=payload["model_state"],
        trainer_state=payload["trainer_state_bytes"], meta_extra=meta,
    )
    print(f"wrote {out / AGENT / ckpt_id}")


if __name__ == "__main__":
    main()
