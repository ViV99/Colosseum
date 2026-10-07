import sys, glob, traceback, copy
sys.path.insert(0, "/home/viv/dev/repos/Colosseum")
import yaml
from colosseum.core.config import load_config, ColosseumConfig
from colosseum.cli import _parse_overrides

R = "/home/viv/dev/repos/Colosseum/configs/examples/"
print("== 1. load example configs")
for f in sorted(glob.glob(R + "*.yaml")):
    try:
        c = load_config(f); print("OK", f.split("/")[-1], "agents=", c.get_trainable_agent_ids())
    except Exception as e:
        print("FAIL", f, e)

base = yaml.safe_load(open(R + "tic_tac_toe.yaml"))

def try_cfg(name, mutate):
    d = copy.deepcopy(base); mutate(d)
    try:
        c = ColosseumConfig.model_validate(d); print(f"[{name}] ACCEPTED"); return c
    except Exception as e:
        print(f"[{name}] REJECTED: {str(e).splitlines()[0:3]}")

print("== 2. typo keys")
c = try_cfg("rollout.num_worker typo", lambda d: d["rollout"].__setitem__("num_worker", 64))
if c: print("   num_workers effective =", c.rollout.num_workers)
try_cfg("top-level 'rollot' typo", lambda d: d.__setitem__("rollot", {"num_workers": 64}))
try_cfg("agents.x.algoritm typo", lambda d: d.__setitem__("agents", {"x": {"algoritm": {"learning_rate": 1}}}))
try_cfg("training.kickstart_teachr typo", lambda d: d["training"].__setitem__("kickstart_teachr", "bc.pt"))
try_cfg("amp_dtype=float8_garbage", lambda d: d["algorithm"].__setitem__("amp_dtype", "garbage"))
try_cfg("learner.device='gpu'", lambda d: d["learner"].__setitem__("device", "gpu"))
try_cfg("recurrent_type='transformer'", lambda d: d["networks"].__setitem__("recurrent_type", "transformer"))
try_cfg("transport.mode=grpc", lambda d: d["transport"].__setitem__("mode", "grpc"))
try_cfg("phase=bc", lambda d: d["training"].__setitem__("phase", "bc"))
try_cfg("num_players=3 for tictactoe", lambda d: d["env"].__setitem__("num_players", 3))
try_cfg("agents with dotted name 'a:b'", lambda d: d.__setitem__("agents", {"a:b": {}}))

print("== 3. per-agent override semantics")
def m(d):
    d["agents"] = {"alpha": None, "beta": {"algorithm": {"learning_rate": 1e-4}}}
c = try_cfg("beta lr only", m)
if c:
    for aid in c.get_trainable_agent_ids():
        a = c.get_agent_config(aid).algorithm
        print(f"   {aid}: lr={a.learning_rate} lr_schedule={a.lr_schedule.value} (global={c.algorithm.lr_schedule.value})")
def m2(d):
    d["agents"] = {"beta": {"networks": {"recurrent_type": "lstm"}}}
try_cfg("beta networks partial (only recurrent_type)", m2)
def m3(d):
    d["agents"] = {"beta": {"learner": {"batch_chunks": 64}}}
c = try_cfg("beta learner.batch_chunks=64", m3)
if c:
    print("   beta learner.device:", c.get_agent_config("beta").learner.device, "queue_size", c.get_agent_config("beta").learner.queue_size, "(global queue_size", c.learner.queue_size, ")")
def m4(d):
    d["agents"] = {"beta": {"training": {"resume_from": "x.pt"}}}
c = try_cfg("beta training.resume_from (not supported field)", m4)

print("== 4. --set override semantics (replicating launcher.run_training)")
def apply(overrides):
    config = ColosseumConfig.model_validate(copy.deepcopy(base))
    od = _parse_overrides(tuple(overrides))
    print("   parsed:", od)
    data = config.model_dump()
    for key, value in od.items():
        parts = key.split("."); d = data
        for p in parts[:-1]: d = d[p]
        d[parts[-1]] = value
    return ColosseumConfig(**data)
for ov in (["rollout.num_worker=16"], ["rollot.num_workers=16"], ["training.resume_from=null"],
           ["training.seed=42"], ["algorithm.learning_rate=1e-4"], ["metrics.use_wandb=True"],
           ["env.kwargs.preset=Round 1"], ["algorithm.lr_schedule=cosine"], ["training.total_timesteps=1_000"],
           ["checkpoint.dir=007"], ["metrics.wandb_project=123"]):
    try:
        c = apply(ov)
        print(f"   {ov} -> OK", {"num_workers": c.rollout.num_workers, "resume_from": c.training.resume_from,
              "lr": c.algorithm.learning_rate, "ckdir": c.checkpoint.dir, "proj": c.metrics.wandb_project, "tt": c.training.total_timesteps})
    except Exception as e:
        print(f"   {ov} -> {type(e).__name__}: {str(e).splitlines()[0]}")
# agents override on yaml with null
b2 = yaml.safe_load(open(R + "tic_tac_toe_multi.yaml"))
config = ColosseumConfig.model_validate(b2)
data = config.model_dump()
try:
    d = data
    for p in "agents.agent_alpha.algorithm".split("."): d = d[p]
    d["learning_rate"] = 1e-4
    print("   agents.agent_alpha.algorithm.learning_rate -> OK")
except Exception as e:
    print(f"   --set agents.agent_alpha.algorithm.learning_rate=1e-4 -> {type(e).__name__}: {e}")
