# backend/ai/move_selector.py

from __future__ import annotations

import random
from typing import Any


def select_strong_move(evaluated_moves: list[dict[str, Any]]) -> dict[str, Any]:
    if not evaluated_moves:
        raise ValueError("候補手がありません")

    return max(evaluated_moves, key=lambda item: item["score"])


def select_level_adjusted_move(
    evaluated_moves: list[dict[str, Any]],
    player_level: float = 0.35,
) -> dict[str, Any]:
    """
    相手の棋力に合わせてAIの強さを変える。

    player_level:
        0.00 = 初心者
        0.35 = 子供・初級者
        0.60 = 中級者
        0.85 = 上級者
        1.00 = ほぼ最善手

    方針:
        弱いAIではなく、
        「その相手にちょうどよい強さ」の自然な手を選ぶ。
    """

    if not evaluated_moves:
        raise ValueError("候補手がありません")

    player_level = max(0.0, min(1.0, float(player_level)))

    moves = sorted(
        evaluated_moves,
        key=lambda item: item["score"],
        reverse=True,
    )

    n = len(moves)

    if n == 1:
        return moves[0]

    # player_level が高いほど上位の手を選ぶ
    # 初心者でも最下位までは落とさない
    min_ratio = 0.05
    max_ratio = 0.75

    target_ratio = max_ratio - (max_ratio - min_ratio) * player_level
    target_index = int(round(target_ratio * (n - 1)))

    # 周辺候補から選ぶことで毎回同じ手を避ける
    window = 2
    start = max(0, target_index - window)
    end = min(n, target_index + window + 1)

    candidates = moves[start:end]

    # ただし、最善手から離れすぎる大悪手は除外
    best_score = moves[0]["score"]

    # 初心者相手なら許容幅広め、強い人相手なら狭め
    allowed_loss = int(1400 - 1000 * player_level)
    allowed_loss = max(250, allowed_loss)

    safe_candidates = [
        item for item in candidates
        if best_score - item["score"] <= allowed_loss
    ]

    if safe_candidates:
        candidates = safe_candidates

    # policyScoreRaw があれば自然な手を優先
    def final_key(item: dict[str, Any]) -> float:
        score_part = item["score"] * 0.01
        policy_part = float(item.get("policyScoreRaw", 0.0)) * 2.0
        noise = random.uniform(-0.3, 0.3)

        return score_part + policy_part + noise

    return max(candidates, key=final_key)

def select_balance_move(
    evaluated_moves: list[dict[str, Any]],
    current_score: int,
    player_level: float = 0.35,
    target_score: int = 0,
) -> dict[str, Any]:
    """
    教育用せめぎ合いAI

    方針:
    - 最善手から少し弱い手を選ぶ
    - ただし自滅手は絶対避ける
    - 相手にチャンスを残す
    """

    if not evaluated_moves:
        raise ValueError("候補手がありません")

    player_level = max(0.0, min(1.0, float(player_level)))

    # 最善応手後の最高評価値
    best_reply_score = max(
        int(item["score"])
        for item in evaluated_moves
    )

    # 最善との差
    # 初心者相手でも致命的悪手は禁止
    allowed_loss = int(700 - 350 * player_level)
    allowed_loss = max(250, allowed_loss)

    safe_moves = [
        item
        for item in evaluated_moves
        if best_reply_score - int(item["score"]) <= allowed_loss
    ]

    if not safe_moves:
        safe_moves = evaluated_moves[:]

    # 0ではなく「少しAI有利」を狙う
    ideal_score = int(250 + 650 * player_level)

    # AIが負けている時は立て直し優先
    if current_score < -300:
        ideal_score = max(
            ideal_score,
            current_score + 700,
        )

    def evaluate(item: dict[str, Any]) -> float:

        # 相手最善応手後
        reply_score = int(item["score"])

        # AIが1手指した直後
        raw_score = int(
            item.get("rawScore", reply_score)
        )

        # 一瞬良いけど次で崩壊する量
        collapse = max(
            0,
            raw_score - reply_score,
        )

        # 現在より悪化
        worsen = max(
            0,
            current_score - reply_score,
        )

        # 目標への近さ
        closeness = -abs(
            reply_score - ideal_score
        )

        # 強さ
        strength = reply_score

        # MCTS探索量
        visit = float(
            item.get("visitCount", 0)
        )

        # policy自然さ
        policy = float(
            item.get("policyScoreRaw", 0.0)
        )

        noise = random.uniform(-0.01, 0.01)

        return (
            strength * 2.0
            + closeness * 0.35
            - collapse * 1.8
            - worsen * 3.0
            + visit * 0.02
            + policy * 0.2
            + noise
        )

    return max(
        safe_moves,
        key=evaluate,
    )

def select_drama_move(
    evaluated_moves: list[dict[str, Any]],
    current_score: int = 0,
) -> dict[str, Any]:
    """
    旧互換用。
    直接50%狙いは弱くなりすぎるので、
    今後は select_level_adjusted_move を使う。
    """

    return select_level_adjusted_move(
        evaluated_moves=evaluated_moves,
        player_level=0.35,
    )

def select_mcts_education_move(
    evaluated_moves: list[dict[str, Any]],
    target_win_rate: float = 55.0,
    player_level: float = 0.35,
) -> dict[str, Any]:
    if not evaluated_moves:
        raise ValueError("候補手がありません")

    player_level = max(0.0, min(1.0, float(player_level)))
    target_win_rate = max(50.0, min(70.0, float(target_win_rate)))

    moves = sorted(
        evaluated_moves,
        key=lambda item: (item.get("visitCount", 0), item["score"]),
        reverse=True,
    )

    best_score = max(item["score"] for item in moves)
    max_visit = max(1, max(int(item.get("visitCount", 0)) for item in moves))

    allowed_loss = int(900 - 500 * player_level)
    allowed_loss = max(250, allowed_loss)

    safe_moves = [
        item for item in moves
        if best_score - int(item["score"]) <= allowed_loss
    ]

    if not safe_moves:
        safe_moves = moves[:]

    def education_key(item: dict[str, Any]) -> float:
        win_rate = float(item.get("winRate", 50.0))
        visit_ratio = float(item.get("visitCount", 0)) / max_visit
        score = float(item.get("score", 0))

        closeness = -abs(win_rate - target_win_rate)
        strength = score / 1200.0
        search_confidence = visit_ratio
        noise = random.uniform(-0.03, 0.03)

        return (
            closeness * (1.25 - 0.45 * player_level)
            + strength * (0.25 + 0.55 * player_level)
            + search_confidence * (0.35 + 0.65 * player_level)
            + noise
        )

    return max(safe_moves, key=education_key)