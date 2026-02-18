"""
FastAPI server that exposes an RL learning endpoint.

Run:
    cd backend
    uvicorn src.server:app --host 0.0.0.0 --port 8000
"""

import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# Make sure sibling modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from rl_trainer import train_on_game  # noqa: E402

app = FastAPI(title="Chess Bot RL Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


class GamePayload(BaseModel):
    moves: list[str]
    result: str
    bot_color: str = "black"

    @field_validator("result")
    @classmethod
    def validate_result(cls, v: str) -> str:
        if v not in ("1-0", "0-1", "1/2-1/2"):
            raise ValueError("result must be '1-0', '0-1', or '1/2-1/2'")
        return v

    @field_validator("bot_color")
    @classmethod
    def validate_color(cls, v: str) -> str:
        if v not in ("white", "black"):
            raise ValueError("bot_color must be 'white' or 'black'")
        return v


@app.post("/api/learn")
async def learn(payload: GamePayload):
    result = train_on_game(
        moves_uci=payload.moves,
        result=payload.result,
        bot_color=payload.bot_color,
    )
    return result
