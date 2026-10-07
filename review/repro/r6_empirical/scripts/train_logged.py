"""Wrapper around colosseum's run_training that configures logging at import time
so spawned learner/worker processes (which re-import __mp_main__) also log to stderr.
usage: train_logged.py CONFIG key=value ..."""
import logging, sys
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(processName)s %(name)s: %(message)s")
if __name__ == "__main__":
    from colosseum.cli import _parse_overrides
    from colosseum.launcher import run_training
    run_training(sys.argv[1], overrides=_parse_overrides(tuple(sys.argv[2:])) or None)
