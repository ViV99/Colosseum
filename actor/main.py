import argparse
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


ID = uuid4()

logger = logging.getLogger(__name__)

app = FastAPI()

with open(pathlib.Path(__file__).parent.resolve() / "config" / "config.yaml") as f:
    config = yaml.safe_load(f)

actor = Actor(TicTacToeEnvironment, config["player_idleness_timeout"], config["max_environments"])
init_arena_client(config["arena_url"], ID)


@app.post("/api/environments/join")
def join_game(player_id: str = Body()):
    logger.info("Player %s want to join new game", player_id)
    try:
        actor.join_game(player_id)
    except ActorError as e:
        raise HTTPException(status_code=e.code, detail=e.type)
    return Response(status_code=204)


@app.get("/api/environments/state")
def get_state(player_id: str):
    logger.info("Player %s requested his state", player_id)
    try:
        return actor.get_state(player_id)
    except ActorError as e:
        raise HTTPException(status_code=e.code, detail=e.type)


@app.post("/api/environments/turn")
def turn(player_id: str = Body(), position: int = Body()):
    logger.info("Player %s want to make turn to position %s", player_id, position)
    try:
        actor.turn(player_id, position)
    except ActorError as e:
        raise HTTPException(status_code=e.code, detail=e.type)
    return Response(status_code=204)


@app.post("/api/ping")
def ping():
    return "PONG"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Actor web server")
    parser.add_argument(
        "host", type=str, help="Host address of web server", default="localhost"
    )
    parser.add_argument("port", type=int, help="Port of web server", default=8000)
    args = parser.parse_args()

    get_arena_client().register_actor(config["max_environments"])

    uvicorn.run("main:app", host=args.host, port=args.port, reload=True)
