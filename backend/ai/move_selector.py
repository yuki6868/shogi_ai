# backend/ai/move_selector.py

from __future__ import annotations

import random
from typing import Any


TARGET_MIN = -300
TARGET_MAX = 300


def select_drama_move(evaluated_moves: list[dict[str, Any]]) -> dict[str, Any]:
    """
    教育用せめぎ合いAI。

    最善手ではなく、
    評価値が -300 ～ +300 に収まる手を優先する。

    + はAI有利
    - は人間有利
    """

    if not evaluated_moves:
        raise ValueError("候補手がありません")

    # 接戦範囲内の手
    drama_moves = [
        item for item in evaluated_moves
        if TARGET_MIN <= item["score"] <= TARGET_MAX
    ]

    if drama_moves:
        return random.choice(drama_moves)

    # 接戦範囲に入らない場合は、0に一番近い手を選ぶ
    return min(evaluated_moves, key=lambda item: abs(item["score"]))


def select_strong_move(evaluated_moves: list[dict[str, Any]]) -> dict[str, Any]:
    """
    比較用。
    AIにとって一番評価値が高い手を選ぶ。
    """
    if not evaluated_moves:
        raise ValueError("候補手がありません")

    return max(evaluated_moves, key=lambda item: item["score"])