# backend/ai/board.py

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Optional


BOARD_SIZE = 9

PROMOTE_MAP = {
    "P": "+P", "L": "+L", "N": "+N", "S": "+S", "B": "+B", "R": "+R",
}

UNPROMOTE_MAP = {v: k for k, v in PROMOTE_MAP.items()}


@dataclass
class Move:
    from_row: Optional[int]
    from_col: Optional[int]
    to_row: int
    to_col: int
    piece: str
    promote: bool = False
    drop: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": None if self.drop else {"row": self.from_row, "col": self.from_col},
            "to": {"row": self.to_row, "col": self.to_col},
            "piece": self.piece,
            "promote": self.promote,
            "drop": self.drop,
        }


class ShogiBoard:
    def __init__(self, board: list[list[Optional[dict[str, Any]]]], turn: str):
        self.board = board
        self.turn = turn  # "player" or "enemy"

    @classmethod
    def from_html_state(cls, data: dict[str, Any]) -> "ShogiBoard":
        return cls(
            board=data["board"],
            turn=data.get("turn", "enemy"),
        )

    def clone(self) -> "ShogiBoard":
        return ShogiBoard(deepcopy(self.board), self.turn)

    def inside(self, row: int, col: int) -> bool:
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def owner_at(self, row: int, col: int) -> Optional[str]:
        piece = self.board[row][col]
        return None if piece is None else piece["owner"]

    def piece_at(self, row: int, col: int) -> Optional[dict[str, Any]]:
        return self.board[row][col]

    def enemy_of(self, owner: str) -> str:
        return "enemy" if owner == "player" else "player"

    def forward(self, owner: str) -> int:
        # playerは上へ進む、enemyは下へ進む想定
        return -1 if owner == "player" else 1

    def can_promote_zone(self, owner: str, row: int) -> bool:
        if owner == "player":
            return row <= 2
        return row >= 6

    def should_offer_promotion(self, owner: str, piece: str, from_row: int, to_row: int) -> bool:
        if piece not in PROMOTE_MAP:
            return False
        return self.can_promote_zone(owner, from_row) or self.can_promote_zone(owner, to_row)

    def must_promote(self, owner: str, piece: str, to_row: int) -> bool:
        if piece in ("P", "L"):
            return to_row == (0 if owner == "player" else 8)
        if piece == "N":
            return to_row in ((0, 1) if owner == "player" else (7, 8))
        return False

    def move_dirs(self, piece: str, owner: str) -> list[tuple[int, int]]:
        f = self.forward(owner)

        gold = [(f, -1), (f, 0), (f, 1), (0, -1), (0, 1), (-f, 0)]

        if piece == "K":
            return [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        if piece == "G":
            return gold
        if piece in ("+P", "+L", "+N", "+S"):
            return gold
        if piece == "S":
            return [(f, -1), (f, 0), (f, 1), (-f, -1), (-f, 1)]
        if piece == "N":
            return [(2 * f, -1), (2 * f, 1)]
        if piece == "P":
            return [(f, 0)]
        if piece == "+B":
            return [(-1, 0), (0, -1), (0, 1), (1, 0)]
        if piece == "+R":
            return [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        return []

    def sliding_dirs(self, piece: str, owner: str) -> list[tuple[int, int]]:
        f = self.forward(owner)

        if piece == "L":
            return [(f, 0)]
        if piece == "B":
            return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        if piece == "R":
            return [(-1, 0), (0, -1), (0, 1), (1, 0)]
        if piece == "+B":
            return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        if piece == "+R":
            return [(-1, 0), (0, -1), (0, 1), (1, 0)]

        return []

    def pseudo_moves_from_square(self, row: int, col: int) -> list[Move]:
        piece_data = self.piece_at(row, col)
        if piece_data is None:
            return []

        owner = piece_data["owner"]
        piece = piece_data["type"]
        moves: list[Move] = []

        for dr, dc in self.move_dirs(piece, owner):
            nr, nc = row + dr, col + dc
            if not self.inside(nr, nc):
                continue
            if self.owner_at(nr, nc) == owner:
                continue

            promote = self.must_promote(owner, piece, nr)
            moves.append(Move(row, col, nr, nc, piece, promote=promote))

            if self.should_offer_promotion(owner, piece, row, nr) and not promote:
                moves.append(Move(row, col, nr, nc, piece, promote=True))

        for dr, dc in self.sliding_dirs(piece, owner):
            nr, nc = row + dr, col + dc
            while self.inside(nr, nc):
                if self.owner_at(nr, nc) == owner:
                    break

                promote = self.must_promote(owner, piece, nr)
                moves.append(Move(row, col, nr, nc, piece, promote=promote))

                if self.should_offer_promotion(owner, piece, row, nr) and not promote:
                    moves.append(Move(row, col, nr, nc, piece, promote=True))

                if self.owner_at(nr, nc) == self.enemy_of(owner):
                    break

                nr += dr
                nc += dc

        return moves

    def generate_pseudo_moves(self, owner: str) -> list[Move]:
        moves: list[Move] = []

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = self.piece_at(r, c)
                if piece and piece["owner"] == owner:
                    moves.extend(self.pseudo_moves_from_square(r, c))

        return moves

    def find_king(self, owner: str) -> Optional[tuple[int, int]]:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = self.piece_at(r, c)
                if piece and piece["owner"] == owner and piece["type"] == "K":
                    return r, c
        return None

    def is_square_attacked(self, row: int, col: int, by_owner: str) -> bool:
        for move in self.generate_pseudo_moves(by_owner):
            if move.to_row == row and move.to_col == col:
                return True
        return False

    def is_in_check(self, owner: str) -> bool:
        king_pos = self.find_king(owner)
        if king_pos is None:
            return True

        kr, kc = king_pos
        return self.is_square_attacked(kr, kc, self.enemy_of(owner))

    def apply_move(self, move: Move) -> None:
        owner = self.turn

        if move.drop:
            self.board[move.to_row][move.to_col] = {
                "type": move.piece,
                "owner": owner,
            }
        else:
            if move.from_row is None or move.from_col is None:
                raise ValueError("通常移動には from_row/from_col が必要です")

            piece_data = self.board[move.from_row][move.from_col]
            if piece_data is None:
                raise ValueError("移動元に駒がありません")

            new_piece = piece_data["type"]
            if move.promote and new_piece in PROMOTE_MAP:
                new_piece = PROMOTE_MAP[new_piece]

            self.board[move.from_row][move.from_col] = None
            self.board[move.to_row][move.to_col] = {
                "type": new_piece,
                "owner": owner,
            }

        self.turn = self.enemy_of(owner)

    def generate_legal_moves(self, owner: Optional[str] = None) -> list[Move]:
        owner = owner or self.turn
        legal: list[Move] = []

        for move in self.generate_pseudo_moves(owner):
            copied = self.clone()
            copied.turn = owner
            copied.apply_move(move)

            if not copied.is_in_check(owner):
                legal.append(move)

        return legal