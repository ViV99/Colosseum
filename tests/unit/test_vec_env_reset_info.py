"""Auto-reset returns the reset info, not the terminal step's info (T3.2, R2-14)."""
import numpy as np
import pytest

from colosseum.envs.subproc_vec_env import SubprocessVectorEnv
from colosseum.envs.vec_env import VectorEnv
from dataflow_helpers import InfoLeakEnv


@pytest.mark.parametrize("kind", ["sync", "subprocess"])
def test_auto_reset_info_is_the_reset_info(kind):
    if kind == "sync":
        vec = VectorEnv(InfoLeakEnv, 1)
    else:
        vec = SubprocessVectorEnv(InfoLeakEnv, 1, num_workers=1)
    try:
        vec.reset_all()
        _, _, terminated, _, infos = vec.step(np.zeros((1, 1), dtype=np.int64))
        assert not terminated[0] and infos[0][0]["phase"] == "step"   # mid-episode: step info
        _, _, terminated, _, infos = vec.step(np.zeros((1, 1), dtype=np.int64))
        assert terminated[0]
        info = infos[0][0]
        assert info["phase"] == "reset" and info["active"] is True
        for stale in ("action_mask", "rank", "outcome"):
            assert stale not in info
        assert info["terminal_info"]["rank"] == 1 and info["terminal_info"]["phase"] == "step"
        assert "terminal_info" not in info["terminal_info"]
        assert info["terminal_observation"][2] == 2.0      # obs = [env_id, ep, t, p] at t=2
    finally:
        vec.close()
