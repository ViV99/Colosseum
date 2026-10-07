from common import *

def dump(tq, title):
    print(f"=== {title}")
    for a, q in tq.items():
        for i, c in enumerate(q.items[:4]):
            print(f" chunk{i} agent={a} obs(ep,t,p)={[tuple(int(x) for x in o[:3]) for o in c.observations]}")
            print(f"   rewards={c.rewards.tolist()} dones={c.dones.tolist()} boot={float(c.bootstrap_value):.3f} ver={c.behavior_policy_version}")

# A: terminated episodes of length 5, chunk 4
tq, rq = run_worker(lambda: StepEnv(L=5), steps=12, chunk_length=4)
dump(tq, "A terminated L=5, T=4")
print(" results:", [(r.player_outcomes, r.episode_length) for r in rq.items])

# B: truncated (time-limit) episodes
tq, rq = run_worker(lambda: StepEnv(L=4, trunc=True), steps=8, chunk_length=4)
dump(tq, "B truncated L=4, T=4 (chunk boundary == truncation)")

# C: turn-based with info['active']; game ends on step 4 (t=4 mover = p0); p1 (loser) inactive at that step
tq, rq = run_worker(lambda: StepEnv(L=5, turn_based=True), steps=20, chunk_length=4)
q = tq["a"].items
p0 = [c for c in q if int(c.observations[0][2]) == 0]
p1 = [c for c in q if int(c.observations[0][2]) == 1]
for name, cs in (("p0", p0), ("p1", p1)):
    for c in cs[:3]:
        print(f" {name} obs(ep,t)={[tuple(int(x) for x in o[:2]) for o in c.observations]} r={c.rewards.tolist()} d={c.dones.tolist()}")
print(" results:", [r.player_outcomes for r in rq.items][:3])
