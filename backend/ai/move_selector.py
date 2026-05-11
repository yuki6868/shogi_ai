# backend/ai/move_selector.py

from __future__ import annotations

from typing import Any


TARGET_SCORE = 150
TARGET_MIN = -300
TARGET_MAX = 500


def select_drama_move(
    evaluated_moves: list[dict[str, Any]],
    current_score: int = 0,
) -> dict[str, Any]:
    """
    教育用せめぎ合いAI。

    ただのランダムではなく、評価値を使って指す。

    current_score:
        現在局面の評価値。
        + ならAI有利
        - ならプレイヤー有利
    """

    if not evaluated_moves:
        raise ValueError("候補手がありません")

    # 1. AIがかなり不利なら、ちゃんと強い手を指す
    if current_score <= -500:
        return max(evaluated_moves, key=lambda item: item["score"])

    # 2. AIが少し不利なら、評価値を改善する手を指す
    if current_score < -100:
        return max(evaluated_moves, key=lambda item: item["score"])

    # 3. AIが有利すぎるなら、勝ちすぎない手を選ぶ
    if current_score >= 900:
        safe_moves = [
            item for item in evaluated_moves
            if TARGET_MIN <= item["score"] <= TARGET_MAX
        ]

        if safe_moves:
            return min(
                safe_moves,
                key=lambda item: abs(item["score"] - TARGET_SCORE),
            )

        return min(
            evaluated_moves,
            key=lambda item: abs(item["score"] - TARGET_SCORE),
        )

    # 4. 互角付近なら、AIが少しだけ良い局面を狙う
    good_drama_moves = [
        item for item in evaluated_moves
        if TARGET_MIN <= item["score"] <= TARGET_MAX
    ]

    if good_drama_moves:
        return min(
            good_drama_moves,
            key=lambda item: abs(item["score"] - TARGET_SCORE),
        )

    # 5. どれも範囲外なら、目標値に一番近い手
    return min(
        evaluated_moves,
        key=lambda item: abs(item["score"] - TARGET_SCORE),
    )


def select_strong_move(evaluated_moves: list[dict[str, Any]]) -> dict[str, Any]:
    """
    評価値最大の手を選ぶ。
    """
    if not evaluated_moves:
        raise ValueError("候補手がありません")

    return max(evaluated_moves, key=lambda item: item["score"])