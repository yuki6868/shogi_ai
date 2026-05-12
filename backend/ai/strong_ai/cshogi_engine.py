from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cshogi

from shogi_core.board import ShogiBoard, Move
from shogi_core.move_encoder import move_to_id


PLAYER_TO_SFEN = {
    "P": "P", "L": "L", "N": "N", "S": "S", "G": "G", "B": "B", "R": "R", "K": "K",
    "+P": "+P", "+L": "+L", "+N": "+N", "+S": "+S", "+B": "+B", "+R": "+R",
}

ENEMY_TO_SFEN = {
    "P": "p", "L": "l", "N": "n", "S": "s", "G": "g", "B": "b", "R": "r", "K": "k",
    "+P": "+p", "+L": "+l", "+N": "+n", "+S": "+s", "+B": "+b", "+R": "+r",
}

HAND_ORDER = ["R", "B", "G", "S", "N", "L", "P"]


@dataclass
class CshogiCandidate:
    move: Move
    move_id: str
    usi: str
    score: float = 0.0
    is_best: bool = False


def _board_to_sfen_board(shogi: ShogiBoard) -> str:
    rows: list[str] = []

    for row in shogi.board:
        empty = 0
        sfen_row = ""

        for cell in row:
            if cell is None:
                empty += 1
                continue

            if empty:
                sfen_row += str(empty)
                empty = 0

            piece = cell["type"]
            owner = cell["owner"]

            sfen_row += PLAYER_TO_SFEN[piece] if owner == "player" else ENEMY_TO_SFEN[piece]

        if empty:
            sfen_row += str(empty)

        rows.append(sfen_row)

    return "/".join(rows)


def _count_hand(hand: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for piece in hand:
        base = piece.replace("+", "")
        counts[base] = counts.get(base, 0) + 1
    return counts


def _hands_to_sfen(shogi: ShogiBoard) -> str:
    parts: list[str] = []

    player_counts = _count_hand(shogi.player_hand)
    enemy_counts = _count_hand(shogi.enemy_hand)

    for piece in HAND_ORDER:
        count = player_counts.get(piece, 0)
        if count:
            parts.append(f"{count if count > 1 else ''}{piece}")

    for piece in HAND_ORDER:
        count = enemy_counts.get(piece, 0)
        if count:
            parts.append(f"{count if count > 1 else ''}{piece.lower()}")

    return "".join(parts) if parts else "-"


def shogi_to_sfen(shogi: ShogiBoard, turn: Optional[str] = None) -> str:
    side = turn or shogi.turn
    board_part = _board_to_sfen_board(shogi)
    turn_part = "b" if side == "player" else "w"
    hand_part = _hands_to_sfen(shogi)
    return f"{board_part} {turn_part} {hand_part} 1"


def shogi_board_to_cshogi_board(shogi: ShogiBoard, turn: Optional[str] = None) -> cshogi.Board:
    return cshogi.Board(shogi_to_sfen(shogi, turn))


def _square_from_usi(square: str) -> tuple[int, int]:
    files = "987654321"
    ranks = "abcdefghi"
    col = files.index(square[0])
    row = ranks.index(square[1])
    return row, col


def usi_to_move(shogi: ShogiBoard, usi: str) -> Move:
    promote = usi.endswith("+")
    body = usi[:-1] if promote else usi

    if "*" in body:
        piece, to_sq = body.split("*")
        to_row, to_col = _square_from_usi(to_sq)

        return Move(
            from_row=None,
            from_col=None,
            to_row=to_row,
            to_col=to_col,
            piece=piece.upper(),
            promote=False,
            drop=True,
        )

    from_sq = body[:2]
    to_sq = body[2:4]

    from_row, from_col = _square_from_usi(from_sq)
    to_row, to_col = _square_from_usi(to_sq)

    piece_data = shogi.board[from_row][from_col]
    if piece_data is None:
        raise ValueError(f"移動元に駒がありません: usi={usi}")

    return Move(
        from_row=from_row,
        from_col=from_col,
        to_row=to_row,
        to_col=to_col,
        piece=piece_data["type"],
        promote=promote,
        drop=False,
    )


def get_cshogi_legal_candidates(shogi: ShogiBoard, turn: str) -> list[CshogiCandidate]:
    cboard = shogi_board_to_cshogi_board(shogi, turn=turn)
    candidates: list[CshogiCandidate] = []

    for cmove in cboard.legal_moves:
        usi = cshogi.move_to_usi(cmove)
        move = usi_to_move(shogi, usi)

        candidates.append(
            CshogiCandidate(
                move=move,
                move_id=move_to_id(move),
                usi=usi,
                score=0.0,
            )
        )

    return candidates