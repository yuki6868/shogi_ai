# backend/main.py

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai.board import ShogiBoard
from ai.evaluator import evaluate_board, evaluate_moves, score_to_win_rate
from ai.move_selector import select_drama_move, select_strong_move


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


@app.post("/api/evaluate")
def evaluate(req: AiMoveRequest):
    shogi = ShogiBoard.from_html_state(req.model_dump())
    score = evaluate_board(shogi, ai_owner="enemy")

    return {
        "ok": True,
        "score": score,
        "aiWinRate": score_to_win_rate(score),
        "playerWinRate": round(100 - score_to_win_rate(score), 1),
        "meaning": "score > 0 ならAI有利、score < 0 なら人間有利です",
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

    evaluated = evaluate_moves(shogi, moves, ai_owner="enemy")
    selected = select_drama_move(evaluated)

    return {
        "ok": True,
        "mode": "drama",
        "move": selected["move"].to_dict(),
        "score": selected["score"],
        "aiWinRate": selected["winRate"],
        "playerWinRate": round(100 - selected["winRate"], 1),
        "legalMoveCount": len(moves),
    }


@app.post("/api/ai-move-strong")
def ai_move_strong(req: AiMoveRequest):
    shogi = ShogiBoard.from_html_state(req.model_dump())
    moves = shogi.generate_legal_moves(req.turn)

    if not moves:
        return {
            "ok": False,
            "reason": "合法手がありません",
            "move": None,
        }

    evaluated = evaluate_moves(shogi, moves, ai_owner="enemy")
    selected = select_strong_move(evaluated)

    return {
        "ok": True,
        "mode": "strong",
        "move": selected["move"].to_dict(),
        "score": selected["score"],
        "aiWinRate": selected["winRate"],
        "playerWinRate": round(100 - selected["winRate"], 1),
        "legalMoveCount": len(moves),
    }


@app.post("/api/check-state")
def check_state(req: AiMoveRequest):
    shogi = ShogiBoard.from_html_state(req.model_dump())

    return {
        "ok": True,
        "turn": req.turn,
        "inCheck": shogi.is_in_check(req.turn),
    }