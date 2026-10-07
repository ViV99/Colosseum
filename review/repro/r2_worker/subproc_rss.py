import sys; sys.path.insert(0, "/home/viv/dev/repos/Colosseum")
from functools import partial
def main():
    from colosseum.launcher import _create_env
    from colosseum.envs.subproc_vec_env import SubprocessVectorEnv
    v = SubprocessVectorEnv(partial(_create_env, "examples.tic_tac_toe.env.TicTacToeEnv", {}), 4, num_workers=2)
    v.reset_all()
    for p in v._procs:
        rss = [l for l in open(f"/proc/{p.pid}/status") if l.startswith("VmRSS")][0].strip()
        print("child", p.pid, rss)
    v.close()
if __name__ == "__main__": main()
