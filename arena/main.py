import pathlib
import logging

import yaml
import uvicorn
from fastapi import FastAPI, Body, Response, Request, Query

from arena.arena_instance import Arena


logger = logging.getLogger(__name__)

app = FastAPI()

arena = Arena()

with open(pathlib.Path(__file__).parent.resolve() / "config" / "config.yaml") as f:
    config = yaml.safe_load(f)


@app.post("/api/actors")
def register_actor(
    request: Request,
    actor_id: str = Body(),
    max_environments: int = Body(),
    max_environment_players: int = Body()
):
    host = request.client.host
    logger.info(
        "Actor from %s has registered with id (%s), max_environments (%s), max_environment_players (%s)",
        host,
        actor_id,
        max_environments,
        max_environment_players
    )
    arena.register_actor(actor_id, host, max_environments, max_environment_players)
    return Response(status_code=204)


@app.delete("/api/environments")
def delete_environment(actor_id: str = Body(), environment_id: str = Body()):
    logger.info("Actor %s reports of environment %s end", actor_id, environment_id)
    arena.delete_environment(actor_id, environment_id)
    return Response(status_code=204)


@app.get("/api/environments")
def get_environment(learner_id: str = Query(), player_id: str = Query()):
    return {"url": arena.get_environment(learner_id, player_id)}


if __name__ == "__main__":
    uvicorn.run("main:app", host=config["host"], port=config["port"])
