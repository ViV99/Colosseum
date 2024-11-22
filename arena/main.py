import logging
import pathlib
import uuid

import uvicorn
import yaml

from fastapi import FastAPI, Body, Response, Request

from arena.arena_instance import Arena


logger = logging.getLogger(__name__)

app = FastAPI()

arena = Arena()

LOCAL_URLS = ["localhost", "127.0.0.1"]


with open(pathlib.Path(__file__).parent.resolve() / "config" / "config.yaml") as f:
    config = yaml.safe_load(f)


@app.post("/api/actors")
def register_actor(request: Request, id: uuid.UUID = Body(), max_environments: int = Body()):
    host = request.client.host
    if host in LOCAL_URLS:
        host = config["host"]

    arena.register_actor(id, host, max_environments)
    return Response(status_code=204)


@app.post("/api/environments")
def add_environment(id: uuid.UUID = Body()):
    arena.add_environment(id)
    return Response(status_code=204)


@app.delete("/api/environments")
def add_environment(id: uuid.UUID = Body()):
    arena.delete_environment(id)
    return Response(status_code=204)


@app.get("/api/actors")
def get_environment_url(learner_id: str, player_id: str):
    return {"url": arena.get_actor_url(learner_id, player_id)}


if __name__ == "__main__":
    uvicorn.run("main:app", host=config["host"], port=config["port"], reload=True)
