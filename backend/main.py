# backend/main.py

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from shogi_core.board import ShogiBoard
from ai.old_ai.evaluator import evaluate_board, evaluate_moves, score_to_win_rate
from ai.old_ai.move_selector import (
    select_balance_move,
    select_level_adjusted_move,
    select_mcts_education_move,
    select_strong_move,
)
from ai.old_ai.mcts import run_mcts, MCTS_SIMULATIONS, MCTS_DEPTH_LIMIT, MAX_CANDIDATES
from shogi_core.move_encoder import legal_moves_to_ids, move_to_id, move_to_dict_with_id
from ai.old_ai.policy_dummy import (
    filter_policy_candidates,
    rank_natural_moves,
    policy_candidates_to_dicts,
)

from ai.old_ai.policy_inference import get_policy_inference
from ai.old_ai.value_inference import get_value_inference

from ai.strong_ai.strong_engine import StrongEngine

from ai.yaneuraou.usi_engine import candidate_to_dict as yaneuraou_candidate_to_dict
from ai.yaneuraou.usi_engine import get_yaneuraou_engine

from ai.competitive.competitive_selector import select_competitive_move


class AiMoveRequest(BaseModel):
    board: list
    playerHand: list | dict = []
    enemyHand: list | dict = []
    turn: str = "enemy"
    playerLevel: float = 0.35
    aiOwner: str = "enemy"

class PlayerMoveReviewRequest(AiMoveRequest):
    playedMoveId: str = ""


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
        "moveIds": legal_moves_to_ids(moves),
        "policyCandidates": [],
        "moves": [move_to_dict_with_id(m) for m in moves],
    }


@app.post("/api/evaluate")
def evaluate(req: AiMoveRequest):
    shogi = ShogiBoard.from_html_state(req.model_dump())
    ai_owner = req.aiOwner if req.aiOwner in ["player", "enemy"] else "enemy"

    score = evaluate_board(shogi, ai_owner=ai_owner)
    ai_win_rate = score_to_win_rate(score)

    return {
        "ok": True,
        "score": score,
        "aiWinRate": ai_win_rate,
        "playerWinRate": round(100 - ai_win_rate, 1),
        "aiOwner": ai_owner,
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
        # ここで1回だけPolicy AI推論する
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

        # もう一度 policy_ai_candidates_to_dicts() を呼ばない
        # ranked_policy の結果をそのまま画面表示用に変換する
        policy_candidates = [
            {
                "moveId": item["moveId"],
                "moveText": move_to_dict_with_id(item["move"]).get("moveText", ""),
                "policyScore": round(float(item["policyScore"]), 4),
                "move": item["move"].to_dict(),
            }
            for item in ranked_policy
        ]

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

        policy_score_map = {
            item["moveId"]: float(item["policyScore"])
            for item in ranked[:12]
        }

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

        "currentScore": current_score,

        "score": display_score,
        "rawScore": display_score,

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

    current_score = evaluate_board(shogi, ai_owner=req.turn)

    # MCTSの深さ探索結果をそのまま使う
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

    selected = select_mcts_education_move(
        evaluated_moves=evaluated,
        target_win_rate=60.0,
        player_level=req.playerLevel,
    )

    # selected = max(
    #     evaluated,
    #     key=lambda item: (
    #         int(item.get("searchScore", item.get("score", -999999))),
    #         int(item.get("rawScore", -999999)),
    #         int(item.get("visitCount", 0)),
    #     ),
    # )

    # 中央表示もMCTS探索後の評価値にする
    display_score = int(selected.get("searchScore", selected["score"]))
    display_win_rate = score_to_win_rate(display_score)

    candidates = [
        {
            "moveId": move_to_id(item["move"]),
            "moveText": move_to_dict_with_id(item["move"]).get("moveText", ""),

            # MCTS深さ探索後の評価
            "score": item.get("searchScore", item["score"]),
            "searchScore": item.get("searchScore", item["score"]),
            "searchWinRate": item.get("searchWinRate", item["winRate"]),

            # AIが1手指した直後の評価
            "rawScore": item.get("rawScore", item["score"]),
            "rawWinRate": item.get("rawWinRate", item["winRate"]),

            "winRate": item["winRate"],
            "visitCount": item.get("visitCount", 0),
            "qValue": item.get("qValue", 0.0),
            "prior": item.get("prior", 0.0),
        }
        for item in evaluated[:8]
    ]

    return {
        "ok": True,
        "mode": "MCTS DEPTH SEARCH + EDUCATION CONTROL",
        "selectedBy": "mcts_education_move",
        "playerLevel": round(req.playerLevel, 3),
        "valueAvailable": get_value_inference().available,

        "moveId": move_to_id(selected["move"]),
        "move": move_to_dict_with_id(selected["move"]),

        "currentScore": current_score,

        # 表示用：MCTS深さ探索後の評価値
        "score": display_score,
        "searchScore": display_score,

        # 参考：AIが1手指した直後
        "rawScore": selected.get("rawScore", display_score),

        "aiWinRate": display_win_rate,
        "playerWinRate": round(100 - display_win_rate, 1),

        "legalMoveCount": len(moves),
        "mctsSimulations": MCTS_SIMULATIONS,
        "mctsDepthLimit": MCTS_DEPTH_LIMIT,
        "maxCandidates": MAX_CANDIDATES,
        "visitCount": selected.get("visitCount", 0),
        "qValue": selected.get("qValue", 0.0),
        "prior": selected.get("prior", 0.0),

        "policyCandidates": candidates,
        "mctsCandidates": candidates,
    }

@app.post("/api/ai-move-strong")
def ai_move_strong(req: AiMoveRequest):

    raw_level = float(req.playerLevel)

    if raw_level >= 0.90:
        level_name = "プロ"
    elif raw_level >= 0.75:
        level_name = "上級"
    elif raw_level >= 0.55:
        level_name = "中級"
    elif raw_level >= 0.35:
        level_name = "初級"
    else:
        level_name = "初心者"

    print("====================================")
    print("AI MOVE STRONG REQUEST")
    print(f"turn         : {req.turn}")
    print(f"aiOwner      : {req.aiOwner}")
    print(f"playerLevel  : {raw_level}")
    print(f"userLevel    : {level_name}")
    print("====================================")
    # print("=== STRONG AI ROUTE: YANEURAOU USI + COMPETITIVE CONTROL ===")

    shogi = ShogiBoard.from_html_state(req.model_dump())
    moves = shogi.generate_legal_moves(req.turn)

    if not moves:
        return {
            "ok": False,
            "reasonCode": "NO_LEGAL_MOVES",
            "reason": "合法手がありません",
            "move": None,
            "mode": "YANEURAOU USI + COMPETITIVE",
            "legalMoveCount": 0,
        }

    engine = get_yaneuraou_engine()

    try:
        candidates = engine.analyze(
            shogi=shogi,
            turn=req.turn,
            depth=8,
            multipv=8,
        )
    except Exception as exc:
        return {
            "ok": False,
            "reasonCode": "ENGINE_ERROR",
            "reason": str(exc),
            "move": None,
            "mode": "YANEURAOU USI + COMPETITIVE",
            "legalMoveCount": len(moves),
        }

    if not candidates:
        return {
            "ok": False,
            "reasonCode": "ENGINE_NO_CANDIDATES",
            "reason": "やねうら王から候補手を取得できませんでした",
            "move": None,
            "mode": "YANEURAOU USI + COMPETITIVE",
            "legalMoveCount": len(moves),
        }

    player_level = max(0.05, min(1.0, float(req.playerLevel)))

    best_score = int(candidates[0].score)

    # playerLevel が低いほど、最善手から大きく落としてもよい
    # playerLevel が高いほど、ほぼ最善手を選ぶ
    max_drop = int(250 + (1.0 - player_level) * 2200)

    # AIが有利なときは、勝ちすぎない評価値を狙う
    # AIが不利なときは、無理に弱い手を選ばず最善寄りにする
    if best_score > 0:
        target_score = int(best_score * player_level)
    else:
        target_score = best_score

    selected = select_competitive_move(
        candidates=candidates,
        target_score=target_score,
        max_drop=max_drop,
    )

    print("----------- COMPETITIVE RESULT -----------")
    print(f"best score      : {candidates[0].score}")
    print(f"selected score  : {selected.score}")
    print(f"selected usi    : {selected.usi}")
    print(f"score drop      : {candidates[0].score - selected.score}")
    print("------------------------------------------")

    for item in candidates:
        item.is_best = item.usi == candidates[0].usi

    selected.is_best = selected.usi == candidates[0].usi

    ai_win_rate = score_to_win_rate(int(selected.score))

    shown_candidates = [
        {
            **yaneuraou_candidate_to_dict(item),
            "selected": item.usi == selected.usi,
            "bestScore": best_score,
            "targetScore": target_score,
            "maxDrop": max_drop,
            "scoreDropFromBest": int(best_score - int(item.score)),
        }
        for item in candidates[:8]
    ]

    return {
        "ok": True,
        "mode": "YANEURAOU USI depth 8 MultiPV 8 + COMPETITIVE CONTROL",
        "selectedBy": "competitive_control",
        "playerLevel": round(player_level, 3),

        "moveId": selected.move_id,
        "move": move_to_dict_with_id(selected.move),

        "score": int(selected.score),
        "searchScore": int(selected.score),
        "rawScore": int(selected.score),

        "aiWinRate": ai_win_rate,
        "playerWinRate": round(100 - ai_win_rate, 1),

        "legalMoveCount": len(moves),
        "engineCandidateCount": len(candidates),
        "enginePath": str(engine.engine_path),

        "bestmoveUsi": candidates[0].usi,
        "selectedUsi": selected.usi,
        "pv": selected.pv[:8],

        "bestScore": best_score,
        "targetScore": target_score,
        "maxDrop": max_drop,
        "scoreDropFromBest": int(best_score - int(selected.score)),
        "isBestMove": selected.is_best,

        "policyCandidates": shown_candidates,
        "mctsCandidates": shown_candidates,
    }

@app.post("/api/review-player-move")
def review_player_move(req: PlayerMoveReviewRequest):
    shogi = ShogiBoard.from_html_state(req.model_dump())
    moves = shogi.generate_legal_moves(req.turn)

    legal_ids = set(legal_moves_to_ids(moves))

    if req.playedMoveId not in legal_ids:
        return {
            "ok": False,
            "reason": "合法手ではありません",
            "playerLevel": round(req.playerLevel, 3),
            "newPlayerLevel": round(req.playerLevel, 3),
            "levelDelta": 0.0,
        }

    player_level = max(0.05, min(1.0, float(req.playerLevel)))

    engine = get_yaneuraou_engine()

    try:
        candidates = engine.analyze(
            shogi=shogi,
            turn=req.turn,
            depth=6,
            multipv=8,
        )
    except Exception as exc:
        return {
            "ok": False,
            "reason": str(exc),
            "playerLevel": round(player_level, 3),
            "newPlayerLevel": round(player_level, 3),
            "levelDelta": 0.0,
        }

    if not candidates:
        return {
            "ok": False,
            "reason": "候補手を取得できませんでした",
            "playerLevel": round(player_level, 3),
            "newPlayerLevel": round(player_level, 3),
            "levelDelta": 0.0,
        }

    candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
    best_score = int(candidates[0].score)

    chosen = None
    chosen_rank = None

    for index, item in enumerate(candidates, start=1):
        if item.move_id == req.playedMoveId:
            chosen = item
            chosen_rank = index
            break

    if chosen is None:
        score_drop = 9999
        chosen_score = None
        level_delta = -0.04
        quality = "候補外"
    else:
        chosen_score = int(chosen.score)
        score_drop = best_score - chosen_score

        if score_drop <= 80:
            level_delta = 0.04
            quality = "最善級"
        elif score_drop <= 180:
            level_delta = 0.025
            quality = "好手"
        elif score_drop <= 350:
            level_delta = 0.0
            quality = "普通"
        elif score_drop <= 700:
            level_delta = -0.025
            quality = "疑問手"
        else:
            level_delta = -0.05
            quality = "悪手"

        if best_score >= 500 and chosen_score < 0:
            level_delta = min(level_delta, -0.06)
            quality = "大きな見落とし"

        if chosen_score <= -800:
            level_delta = min(level_delta, -0.08)
            quality = "大悪手"

    move_count = len(req.moveHistory) if hasattr(req, "moveHistory") and req.moveHistory else 0

    if move_count < 12:
        level_delta *= 0.15
        opening_phase = "序盤のためレベル変動をかなり抑制"
    elif move_count < 24:
        level_delta *= 0.4
        opening_phase = "序盤〜中盤のためレベル変動を抑制"
    else:
        opening_phase = "通常のレベル変動"

    new_level = max(0.05, min(1.0, player_level + level_delta))

    return {
        "ok": True,
        "playerLevel": round(player_level, 3),
        "newPlayerLevel": round(new_level, 3),
        "levelDelta": round(level_delta, 3),
        "openingPhase": opening_phase,
        "moveCount": move_count,
        "quality": quality,
        "playedMoveId": req.playedMoveId,
        "chosenRank": chosen_rank,
        "bestScore": best_score,
        "chosenScore": chosen_score,
        "scoreDrop": score_drop,
        "candidates": [
            {
                **yaneuraou_candidate_to_dict(item),
                "selected": item.move_id == req.playedMoveId,
                "scoreDropFromBest": int(best_score - int(item.score)),
            }
            for item in candidates[:8]
        ],
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