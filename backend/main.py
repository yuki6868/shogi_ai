# backend/main.py

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai.board import ShogiBoard
from ai.evaluator import evaluate_board, evaluate_moves, score_to_win_rate
from ai.move_selector import (
    select_balance_move,
    select_level_adjusted_move,
    select_mcts_education_move,
    select_strong_move,
)
from ai.mcts import run_mcts, MCTS_SIMULATIONS, MCTS_DEPTH_LIMIT, MAX_CANDIDATES
from ai.move_encoder import legal_moves_to_ids, move_to_id, move_to_dict_with_id
from ai.policy_dummy import (
    filter_policy_candidates,
    rank_natural_moves,
    policy_candidates_to_dicts,
)

from ai.policy_inference import (
    get_policy_inference,
    filter_policy_ai_candidates,
    policy_ai_candidates_to_dicts,
)
from ai.value_inference import get_value_inference


class AiMoveRequest(BaseModel):
    board: list
    playerHand: list | dict = []
    enemyHand: list | dict = []
    turn: str = "enemy"
    playerLevel: float = 0.35


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
        "moveIds": legal_moves_to_ids(moves),
        "policyCandidates": policy_candidates_to_dicts(ranked, limit=12),
        "moves": [move_to_dict_with_id(m) for m in moves],
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

    policy = get_policy_inference()

    policy_score_map = {}

    if policy.available:
        ranked_policy = policy.rank_legal_moves(
            shogi=shogi,
            legal_moves=moves,
            top_k=12,
        )

        policy_moves = [item["move"] for item in ranked_policy]
        policy_score_map = {
            item["moveId"]: float(item["policyScore"])
            for item in ranked_policy
        }

        policy_candidates = policy_ai_candidates_to_dicts(
            shogi=shogi,
            legal_moves=moves,
            limit=12,
        )
        policy_mode = "POLICY AI"
    else:
        policy_moves = filter_policy_candidates(
            shogi,
            moves,
            owner=req.turn,
            top_k=12,
        )
        ranked = rank_natural_moves(shogi, moves, req.turn)
        policy_candidates = policy_candidates_to_dicts(ranked, limit=12)
        policy_mode = "POLICY DUMMY"

    evaluated = evaluate_moves(
        shogi,
        policy_moves,
        ai_owner="enemy",
        lookahead=True,
    )

    for item in evaluated:
        item["policyScoreRaw"] = policy_score_map.get(
            move_to_id(item["move"]),
            0.0,
        )

    selected = select_balance_move(
        evaluated_moves=evaluated,
        current_score=current_score,
        player_level=req.playerLevel,
        target_score=0,
    )

    display_score = selected.get("rawScore", selected["score"])
    display_win_rate = score_to_win_rate(display_score)

    return {
        "ok": True,
        "mode": f"LEVEL ADJUST + {policy_mode}",
        "valueAvailable": get_value_inference().available,
        "selectedBy": "balance_policy_value",
        "playerLevel": round(req.playerLevel, 3),

        "moveId": move_to_id(selected["move"]),
        "move": move_to_dict_with_id(selected["move"]),

        # 現局面
        "currentScore": current_score,

        # 画面表示用：AIが一手指した直後の評価値
        "score": display_score,
        "rawScore": display_score,

        # 内部選択用：相手の最善応手込みの評価値
        "replyScore": selected["score"],

        "aiWinRate": display_win_rate,
        "playerWinRate": round(100 - display_win_rate, 1),

        "legalMoveCount": len(moves),
        "policyMoveCount": len(policy_moves),
        "policyCandidates": [
            {
                **item,
                "note": "policyScoreは評価値ではなく、自然さスコアです",
            }
            for item in policy_candidates
        ],
    }

@app.post("/api/ai-move-mcts")
def ai_move_mcts(req: AiMoveRequest):
    shogi = ShogiBoard.from_html_state(req.model_dump())
    shogi.turn = req.turn

    moves = shogi.generate_legal_moves(req.turn)

    if not moves:
        return {
            "ok": False,
            "reason": "合法手がありません",
            "move": None,
        }

    current_score = evaluate_board(shogi, ai_owner="enemy")

    evaluated = run_mcts(
        shogi=shogi,
        root_owner=req.turn,
        simulations=MCTS_SIMULATIONS,
        depth_limit=MCTS_DEPTH_LIMIT,
    )

    if not evaluated:
        return {
            "ok": False,
            "reason": "MCTS候補手がありません",
            "move": None,
        }

    selected = select_balance_move(
        evaluated_moves=evaluated,
        current_score=current_score,
        player_level=req.playerLevel,
        target_score=0,
    )

    display_score = selected.get("rawScore", selected["score"])
    display_win_rate = score_to_win_rate(display_score)

    candidates = [
        {
            "moveId": move_to_id(item["move"]),
            "moveText": move_to_dict_with_id(item["move"]).get("moveText", ""),
            "score": item["score"],
            "rawScore": item.get("rawScore", item["score"]),
            "winRate": item["winRate"],
            "searchScore": item.get("searchScore", item["score"]),
            "searchWinRate": item.get("searchWinRate", item["winRate"]),
            "visitCount": item.get("visitCount", 0),
            "qValue": item.get("qValue", 0.0),
            "prior": item.get("prior", 0.0),
        }
        for item in evaluated[:8]
    ]

    return {
        "ok": True,
        "mode": "LIGHT MCTS + POLICY + VALUE + EDUCATION CONTROL",
        "selectedBy": "mcts_education_target_55",
        "playerLevel": round(req.playerLevel, 3),
        "valueAvailable": get_value_inference().available,

        "moveId": move_to_id(selected["move"]),
        "move": move_to_dict_with_id(selected["move"]),

        "currentScore": current_score,

        # 画面表示用：AIが1手指した直後の評価値
        "score": display_score,
        "rawScore": display_score,

        # MCTS探索スコアは別名で返す
        "searchScore": selected.get("searchScore", selected["score"]),

        "aiWinRate": display_win_rate,
        "playerWinRate": round(100 - display_win_rate, 1),

        "legalMoveCount": len(moves),
        "mctsSimulations": MCTS_SIMULATIONS,
        "mctsDepthLimit": MCTS_DEPTH_LIMIT,
        "maxCandidates": MAX_CANDIDATES,
        "visitCount": selected.get("visitCount", 0),
        "qValue": selected.get("qValue", 0.0),
        "prior": selected.get("prior", 0.0),

        # フロントが見ている名前に合わせる
        "policyCandidates": candidates,

        # デバッグ用に残す
        "mctsCandidates": candidates,
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

@app.get("/api/policy-status")
def policy_status():
    policy = get_policy_inference()

    return {
        "ok": True,
        "available": policy.available,
        "modelPath": str(policy.model_path),
        "device": str(policy.device),
    }

@app.get("/api/value-status")
def value_status():
    value_ai = get_value_inference()

    return {
        "ok": True,
        "available": value_ai.available,
        "modelPath": str(value_ai.model_path),
        "device": str(value_ai.device),
    }