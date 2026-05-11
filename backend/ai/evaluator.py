# backend/ai/evaluator.py

from __future__ import annotations

import math
from typing import Any

from ai.board import ShogiBoard, Move


PIECE_VALUES = {
    "P": 100,
    "L": 300,
    "N": 300,
    "S": 500,
    "G": 600,
    "B": 800,
    "R": 1000,
    "K": 0,

    "+P": 600,
    "+L": 600,
    "+N": 600,
    "+S": 600,
    "+B": 1100,
    "+R": 1300,
}


def owner_sign(owner: str, ai_owner: str = "enemy") -> int:
    return 1 if owner == ai_owner else -1


def evaluate_board(shogi: ShogiBoard, ai_owner: str = "enemy") -> int:
    """
    評価値を返す。
    + ならAI有利
    - なら人間有利
    0 なら互角
    """
    score = 0

    # 盤上の駒得
    for r in range(9):
        for c in range(9):
            piece = shogi.board[r][c]
            if piece is None:
                continue

            piece_type = piece["type"]
            owner = piece["owner"]

            value = PIECE_VALUES.get(piece_type, 0)
            score += owner_sign(owner, ai_owner) * value

            # 敵陣に近い駒を少し評価
            score += owner_sign(owner, ai_owner) * position_bonus(owner, piece_type, r, c)

    # 持ち駒評価
    for piece in shogi.enemy_hand:
        score += PIECE_VALUES.get(piece, 0) * 0.9

    for piece in shogi.player_hand:
        score -= PIECE_VALUES.get(piece, 0) * 0.9

    # 王の安全度
    score += king_safety_score(shogi, ai_owner)
    score -= king_safety_score(shogi, shogi.enemy_of(ai_owner))

    return int(score)


def position_bonus(owner: str, piece_type: str, row: int, col: int) -> int:
    """
    簡易的な位置評価。
    前に進んでいる駒、中央にいる駒を少し評価する。
    """
    bonus = 0

    # 中央に近いほど少し良い
    center_distance = abs(row - 4) + abs(col - 4)
    bonus += max(0, 20 - center_distance * 3)

    # 敵陣に近いほど少し良い
    if owner == "enemy":
        bonus += row * 5
    else:
        bonus += (8 - row) * 5

    # 成り駒は前線にいると強い
    if piece_type.startswith("+"):
        bonus += 40

    return bonus


def king_safety_score(shogi: ShogiBoard, owner: str) -> int:
    """
    王の周辺に自駒が多いほど安全とみなす。
    """
    king_pos = shogi.find_king(owner)
    if king_pos is None:
        return -99999

    kr, kc = king_pos
    score = 0

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue

            nr = kr + dr
            nc = kc + dc

            if not shogi.inside(nr, nc):
                continue

            piece = shogi.board[nr][nc]
            if piece is None:
                continue

            if piece["owner"] == owner:
                score += 40
            else:
                score -= 60

    if shogi.is_in_check(owner):
        score -= 500

    return score


def evaluate_move(shogi: ShogiBoard, move: Move, ai_owner: str = "enemy") -> dict[str, Any]:
    copied = shogi.clone()
    copied.turn = shogi.turn
    copied.apply_move(move)

    score = evaluate_board(copied, ai_owner=ai_owner)

    return {
        "move": move,
        "score": score,
        "winRate": score_to_win_rate(score),
    }


def evaluate_moves(shogi: ShogiBoard, moves: list[Move], ai_owner: str = "enemy") -> list[dict[str, Any]]:
    return [evaluate_move(shogi, move, ai_owner=ai_owner) for move in moves]


def score_to_win_rate(score: int, k: int = 600) -> float:
    """
    評価値を勝率っぽい数字に変換する。
    score = 0 なら 50%
    """
    p = 1 / (1 + math.exp(-score / k))
    return round(p * 100, 1)