from common import *
import multiprocessing as mp, time
from colosseum.core.types import WorkerCommand
from colosseum.worker.rollout_worker import _drain_commands, _apply_command
# Two refreshes queue up while the worker is blocked (e.g. on a full trajectory queue).
q = mp.Queue(maxsize=2)
sd = make_net().state_dict()
q.put(WorkerCommand(slot_agent_map=[["a","a"]], slot_network_map=[["latest","ckpt_v100"]], collect_mask=[[True,False]], new_checkpoints={"a": {"ckpt_v100": sd}}))
q.put(WorkerCommand(slot_agent_map=[["a","a"]], slot_network_map=[["latest","ckpt_v100"]], collect_mask=[[True,False]], new_checkpoints={}))
time.sleep(0.3)
nets = {"a": {"latest": make_net()}}; pending = {}
cmd = _drain_commands(q); _apply_command(cmd, nets, {"a": make_net}, pending)
print("worker pool after draining 2 commands:", list(nets["a"].keys()), "; staged slot nets:", pending["slot_network_map"])
# 3rd refresh: launcher would mark ckpt_v100 as already sent (worker_broadcast_ckpts) -> never re-sent.
# Show the worker silently plays 'latest' in a slot labelled ckpt_v100 and reports it as such:
rq = ListQ()
tq = {"a": ListQ()}; 
rollout_worker_process(0, lambda: StepEnv(L=3), 1, 4, ["a"], {"a": make_net}, tq, {"a": ListQ()}, threading.Event(),
    total_timesteps=3, slot_agent_map=[["a","a"]], slot_network_map=[["latest","ckpt_v100"]], collect_mask=[[True,False]], results_queue=rq)
print("no crash, result keys reported:", list(rq.items[0].player_outcomes.keys()), "<- slot 1 actually ran 'latest' weights")
# Scripted / unknown agent in a slot
try:
    rollout_worker_process(0, lambda: StepEnv(L=3), 1, 4, ["a"], {"a": make_net}, {"a": ListQ()}, {"a": ListQ()}, threading.Event(),
        total_timesteps=3, slot_agent_map=[["a","scripted_bot"]], collect_mask=[[True,False]])
    print("scripted slot OK")
except Exception as e:
    print("scripted/frozen-external agent in slot ->", type(e).__name__, e)
