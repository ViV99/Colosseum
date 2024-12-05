import os
import logging
import pathlib
from uuid import uuid4

import uvicorn
import yaml

from fastapi import FastAPI, Body, HTTPException, Response

from actor.actor_instance import Actor
from actor.arena_client import get_arena_client, init_arena_client
from actor.environments import TicTacToeEnvironment
from actor.error import ActorError


ID = str(uuid4())

logger = logging.getLogger(__name__)

app = FastAPI()

with open(pathlib.Path(__file__).parent.resolve() / "config" / "config.yaml") as f:
    config = yaml.safe_load(f)


actor = Actor(TicTacToeEnvironment, config["player_idleness_timeout"], config["max_environments"])
init_arena_client(os.getenv("ARENA_URL", "http://arena-service:8000"), ID)


@app.post("/api/environments/create")
def create_environment(player_ids: list[str] = Body()):
    logger.info("Matchmaker wants to create new env %s for players %s", id, player_ids)
    try:
        return Response(status_code=201, content={"environment_id": actor.create_environment(player_ids)})
    except ActorError as e:
        raise HTTPException(status_code=e.code, detail=e.type)


@app.get("/api/environments/state")
def get_state(player_id: str):
    logger.info("Player %s requested his state", player_id)
    try:
        return actor.get_state(player_id)
    except ActorError as e:
        raise HTTPException(status_code=e.code, detail=e.type)


@app.post("/api/environments/turn")
def turn(player_id: str = Body(), turn_info: dict = Body()):
    logger.info("Player %s wants to make turn: %s", player_id, turn_info)
    try:
        actor.turn(player_id, **turn_info)
    except ActorError as e:
        raise HTTPException(status_code=e.code, detail=e.type)
    return Response(status_code=204)


@app.post("/api/ping")
def ping():
    return "PONG"


if __name__ == "__main__":
    get_arena_client().register_actor(actor.max_environments, actor.max_environment_players)

    uvicorn.run("main:app", host=config["host"], port=config["port"])
