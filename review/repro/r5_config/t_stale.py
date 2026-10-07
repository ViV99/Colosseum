"""Stale checkpoint dir reuse across different games / architectures."""
import sys, shutil
sys.path.insert(0, "/home/viv/dev/repos/Colosseum")
from colosseum.core.config import load_config
from colosseum.core.registry import build_network
from colosseum.coordinator.coordinator import Coordinator
from colosseum.launcher import _derive_worker_configs
R = "/home/viv/dev/repos/Colosseum/configs/examples/"
shutil.rmtree("ck_shared", ignore_errors=True)
ttt = load_config(R + "tic_tac_toe.yaml"); ttt.checkpoint.dir = "ck_shared"
co = Coordinator(ttt)
co.checkpoint_manager.save("agent_0", 50, {k: v.clone() for k, v in build_network(ttt).state_dict().items()})
print("saved ttt ckpt for agent_0")
chase = load_config(R + "chase.yaml"); chase.checkpoint.dir = "ck_shared"
co2 = Coordinator(chase); co2.agent_pool.register_trainable("agent_0")
mc = co2.generate_match_configs("agent_0", 8)
ck, snm, cm, sam = _derive_worker_configs(mc, co2, ["agent_0"])
print("slot_network_map:", snm[:3])
for cid, sd in ck["agent_0"].items():
    net = build_network(chase)
    try:
        net.load_state_dict(sd); print("load OK", cid)
    except Exception as e:
        print("WORKER WOULD CRASH loading", cid, ":", str(e).splitlines()[0][:200])
