# backend/ai/competitive/competitive_selector.py

from __future__ import annotations

from typing import Any


def select_competitive_move(
    candidates: list[Any],
    target_score: float = 0.0,
    max_drop: float = 300.0,
) -> Any:
    """
    強いAIの候補手から、せめぎ合い用の手を選ぶ。

    candidates:
      score 属性を持つ候補手リスト

    max_drop:
      最善手から何点まで落としてよいか。
      例: 300なら、最善手より300点以内の手だけ選ぶ。
    """

    if not candidates:
        raise ValueError("候補手がありません")

    candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
    best_score = candidates[0].score

    safe_moves = [
        move for move in candidates
        if best_score - move.score <= max_drop
    ]

    if not safe_moves:
        return candidates[0]

    return min(safe_moves, key=lambda x: abs(x.score - target_score))
