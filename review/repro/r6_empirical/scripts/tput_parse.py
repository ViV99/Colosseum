import re, sys, datetime as dt
def ts(l): return dt.datetime.strptime(l[:23], "%Y-%m-%d %H:%M:%S,%f").timestamp()
for f in sys.argv[1:]:
    L = open(f).read().splitlines()
    starts = {}; steps = {}; sent = {}; lsteps = []; recv = []
    for l in L:
        m = re.search(r"Worker (\d+): starting", l)
        if m: starts[m.group(1)] = ts(l)
        m = re.search(r"Worker (\d+): finished. Steps=(\d+), chunks_sent=(\d+)", l)
        if m: steps[m.group(1)] = (ts(l), int(m.group(2))); sent[m.group(1)] = int(m.group(3))
        m = re.search(r"Learner \[\w+\]: step=(\d+), chunks=(\d+)", l)
        if m: lsteps.append((ts(l), int(m.group(1)), int(m.group(2))))
    env_rate = sum(s / (t - starts[k]) for k, (t, s) in steps.items() if k in starts)
    if len(lsteps) > 2:
        (t0, s0, c0), (t1, s1, c1) = lsteps[1], lsteps[-1]
        upd = (s1 - s0) / (t1 - t0); chunk_rate = (c1 - c0) / (t1 - t0); last_c = c1
    else:
        upd = chunk_rate = 0; last_c = lsteps[-1][2] if lsteps else 0
    print(f"{f.split('/')[-1]:28s} workers={len(steps)} env_steps/s={env_rate:8.0f} learner_upd/s={upd:6.2f} "
          f"chunks/s={chunk_rate:7.1f} chunks_sent={sum(sent.values())} chunks_recv~={last_c}")
