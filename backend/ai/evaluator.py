# backend/ai/evaluator.py

from __future__ import annotations

import math
from typing import Any

from ai.board import ShogiBoard, Move
from ai.value_inference import get_value_inference


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
    value_ai = get_value_inference()

    if value_ai.available:
        return value_ai.evaluate_score(shogi, ai_owner=ai_owner)

    score = 0

    for r in range(9):
        for c in range(9):
            piece = shogi.board[r][c]
            if piece is None:
                continue

            piece_type = piece["type"]
            owner = piece["owner"]

            value = PIECE_VALUES.get(piece_type, 0)

            score += owner_sign(owner, ai_owner) * value
            score += owner_sign(owner, ai_owner) * position_bonus(
                owner,
                piece_type,
                r,
                c,
            )

    for piece in shogi.enemy_hand:
        score += int(PIECE_VALUES.get(piece, 0) * 0.9)

    for piece in shogi.player_hand:
        score -= int(PIECE_VALUES.get(piece, 0) * 0.9)

    score += king_safety_score(shogi, ai_owner)
    score -= king_safety_score(shogi, shogi.enemy_of(ai_owner))

    return int(score)


def position_bonus(owner: str, piece_type: str, row: int, col: int) -> int:
    bonus = 0

    center_distance = abs(row - 4) + abs(col - 4)
    bonus += max(0, 20 - center_distance * 3)

    if owner == "enemy":
        bonus += row * 5
    else:
        bonus += (8 - row) * 5

    if piece_type.startswith("+"):
        bonus += 40

    return bonus


def king_safety_score(shogi: ShogiBoard, owner: str) -> int:
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


def score_to_win_rate(score: int, k: int = 600) -> float:
    p = 1 / (1 + math.exp(-score / k))
    return round(p * 100, 1)


def evaluate_move_one_ply(
    shogi: ShogiBoard,
    move: Move,
    ai_owner: str = "enemy",
) -> dict[str, Any]:
    """
    AIが1手指した直後の評価。
    """
    copied = shogi.clone()
    copied.apply_move(move)

    score = evaluate_board(copied, ai_owner=ai_owner)

    return {
        "move": move,
        "score": score,
        "rawScore": score,
        "winRate": score_to_win_rate(score),
    }


def evaluate_move_with_reply(
    shogi: ShogiBoard,
    move: Move,
    ai_owner: str = "enemy",
) -> dict[str, Any]:
    """
    AIが1手指す
    ↓
    プレイヤーが一番AIにとって嫌な手を指す
    ↓
    その後の評価値を見る

    これで、ただ駒を取るだけの弱いAIよりかなりマシになる。
    """
    after_ai = shogi.clone()
    after_ai.apply_move(move)

    raw_score = evaluate_board(after_ai, ai_owner=ai_owner)

    player_owner = after_ai.enemy_of(ai_owner)
    replies = after_ai.generate_legal_moves(player_owner)

    if not replies:
        return {
            "move": move,
            "score": raw_score + 5000,
            "rawScore": raw_score,
            "winRate": score_to_win_rate(raw_score + 5000),
        }

    worst_score_for_ai = 999999

    for reply in replies:
        after_player = after_ai.clone()
        after_player.apply_move(reply)

        reply_score = evaluate_board(after_player, ai_owner=ai_owner)

        if reply_score < worst_score_for_ai:
            worst_score_for_ai = reply_score

    return {
        "move": move,
        "score": int(worst_score_for_ai),
        "rawScore": int(raw_score),
        "winRate": score_to_win_rate(int(worst_score_for_ai)),
    }


def evaluate_moves(
    shogi: ShogiBoard,
    moves: list[Move],
    ai_owner: str = "enemy",
    lookahead: bool = True,
) -> list[dict[str, Any]]:
    if lookahead:
        return [
            evaluate_move_with_reply(shogi, move, ai_owner=ai_owner)
            for move in moves
        ]

    return [
        evaluate_move_one_ply(shogi, move, ai_owner=ai_owner)
        for move in moves
    ]