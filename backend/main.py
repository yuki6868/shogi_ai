# backend/main.py

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai.board import ShogiBoard
from ai.evaluator import evaluate_board, evaluate_moves, score_to_win_rate
from ai.move_selector import select_drama_move, select_strong_move
from ai.move_encoder import legal_moves_to_ids, move_to_id, move_to_dict_with_id
from ai.policy_dummy import filter_policy_candidates, rank_natural_moves


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

    ranked = rank_natural_moves(shogi, moves, req.turn)

    return {
        "ok": True,
        "turn": req.turn,
        "count": len(moves),
        "policyCandidates": ranked[:12],
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

    current_score = evaluate_board(shogi, ai_owner="enemy")

    policy_moves = filter_policy_candidates(
        shogi,
        moves,
        owner=req.turn,
        top_k=12,
    )

    evaluated = evaluate_moves(
        shogi,
        policy_moves,
        ai_owner="enemy",
        lookahead=True,
    )

    selected = select_drama_move(
        evaluated,
        current_score=current_score,
    )

    return {
        "ok": True,
        "mode": "DRAMA THINK",
        "moveId": move_to_id(selected["move"]),
        "move": move_to_dict_with_id(selected["move"]),
        "currentScore": current_score,
        "score": selected["score"],
        "rawScore": selected.get("rawScore", selected["score"]),
        "aiWinRate": selected["winRate"],
        "playerWinRate": round(100 - selected["winRate"], 1),
        "legalMoveCount": len(moves),
        "policyMoveCount": len(policy_moves),
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

    evaluated = evaluate_moves(
        shogi,
        moves,
        ai_owner="enemy",
        lookahead=True,
    )

    selected = select_strong_move(evaluated)

    return {
        "ok": True,
        "mode": "STRONG THINK",
        "moveId": move_to_id(selected["move"]),
        "move": move_to_dict_with_id(selected["move"]),
        "score": selected["score"],
        "rawScore": selected.get("rawScore", selected["score"]),
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