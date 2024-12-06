import os
import logging
import pathlib

import yaml

from learner.clients import init_arena_client
from learner.learner_instance import Learner
from learner.models import IModel

logger = logging.getLogger(__name__)

with open(pathlib.Path(__file__).parent.resolve() / "config" / "config.yaml") as f:
    config = yaml.safe_load(f)

init_arena_client(os.getenv("ARENA_URL", "http://arena-service:8000"))


if __name__ == "__main__":
    learner = Learner(4, 228, IModel)
    learner.run()
