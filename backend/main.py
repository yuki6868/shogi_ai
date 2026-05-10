# backend/main.py

from __future__ import annotations

import random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai.board import ShogiBoard


class AiMoveRequest(BaseModel):
    board: list
    playerHand: list | dict = []
    enemyHand: list | dict = []
    turn: str = "enemy"


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/api/legal-moves")
def legal_moves(req: AiMoveRequest):
    shogi = ShogiBoard.from_html_state(req.model_dump())
    moves = shogi.generate_legal_moves(req.turn)

    return {
        "ok": True,
        "turn": req.turn,
        "count": len(moves),
        "moves": [m.to_dict() for m in moves],
    }


@app.post("/api/ai-move")
def ai_move(req: AiMoveRequest):
    shogi = ShogiBoard.from_html_state(req.model_dump())
    moves = shogi.generate_legal_moves(req.turn)

    if not moves:
        return {
            "ok": False,
            "reason": "合法手がありません",
            "move": None,
        }

    selected = random.choice(moves)

    return {
        "ok": True,
        "move": selected.to_dict(),
        "legalMoveCount": len(moves),
    }