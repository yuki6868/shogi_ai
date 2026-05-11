# backend/ai/policy_dummy.py

from __future__ import annotations

from ai.board import Move, ShogiBoard
from ai.move_encoder import move_to_id


PIECE_VALUES = {
    "P": 100,
    "L": 300,
    "N": 300,
    "S": 500,
    "G": 600,
    "B": 800,
    "R": 1000,
    "K": 10000,
    "+P": 600,
    "+L": 600,
    "+N": 600,
    "+S": 600,
    "+B": 1100,
    "+R": 1300,
}


def score_natural_move(shogi: ShogiBoard, move: Move, owner: str) -> float:
    score = 0.0

    # 成りは自然
    if move.promote:
        score += 180

    # 駒を取る手は自然
    captured = shogi.board[move.to_row][move.to_col]
    if captured is not None and captured["owner"] != owner:
        score += PIECE_VALUES.get(captured["type"], 0) * 0.8

    # 王手っぽい手は少し評価
    copied = shogi.clone()
    copied.turn = owner
    try:
        copied.apply_move(move)
        if copied.is_in_check(copied.enemy_of(owner)):
            score += 250
    except Exception:
        pass

    # 中央に近い手は自然
    center_distance = abs(move.to_row - 4) + abs(move.to_col - 4)
    score += max(0, 40 - center_distance * 6)

    # 前進する手を少し評価
    if not move.drop and move.from_row is not None:
        if owner == "enemy":
            score += max(0, move.to_row - move.from_row) * 30
        else:
            score += max(0, move.from_row - move.to_row) * 30

    # 駒打ちはそれなりに自然
    if move.drop:
        score += 80

        # 歩打ちは自然だが、打ちすぎを少し抑える
        if move.piece == "P":
            score += 20
        else:
            score += 60

    # 駒の種類による軽い補正
    score += PIECE_VALUES.get(move.piece, 0) * 0.03

    return score


def rank_natural_moves(
    shogi: ShogiBoard,
    legal_moves: list[Move],
    owner: str,
) -> list[dict]:
    ranked = []

    for move in legal_moves:
        ranked.append(
            {
                "move": move,
                "moveId": move_to_id(move),
                "policyScore": score_natural_move(shogi, move, owner),
            }
        )

    ranked.sort(key=lambda item: item["policyScore"], reverse=True)
    return ranked


def filter_policy_candidates(
    shogi: ShogiBoard,
    legal_moves: list[Move],
    owner: str,
    top_k: int = 12,
) -> list[Move]:
    if not legal_moves:
        return []

    ranked = rank_natural_moves(shogi, legal_moves, owner)
    candidates = [item["move"] for item in ranked[:top_k]]

    return candidates if candidates else legal_moves