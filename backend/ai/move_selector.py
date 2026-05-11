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