"""Learner pushes with put_nowait to a maxsize=2 mp.Queue (launcher._WEIGHT_QUEUE_SIZE);
worker drains with get_nowait loop every weight_sync_interval (rollout_worker._sync_weights)."""
import multiprocessing as mp, queue, time
q = mp.Queue(maxsize=2)
received = []
for v in range(1, 51):            # learner pushes v=1..50 between two worker syncs
    try: q.put_nowait(v)
    except queue.Full: pass
time.sleep(0.2)
latest = None
while True:
    try: latest = q.get_nowait()
    except queue.Empty: break
print("learner latest version = 50; worker loaded version =", latest)
