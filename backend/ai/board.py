# backend/ai/board.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


BOARD_SIZE = 9

PROMOTE_MAP = {
    "P": "+P",
    "L": "+L",
    "N": "+N",
    "S": "+S",
    "B": "+B",
    "R": "+R",
}

UNPROMOTE_MAP = {v: k for k, v in PROMOTE_MAP.items()}

PIECE_MAP = {
    "歩": "P",
    "香": "L",
    "桂": "N",
    "銀": "S",
    "金": "G",
    "角": "B",
    "飛": "R",
    "王": "K",
    "玉": "K",
    "と": "+P",
    "成香": "+L",
    "成桂": "+N",
    "成銀": "+S",
    "馬": "+B",
    "龍": "+R",
}

REVERSE_PIECE_MAP = {
    "P": "歩",
    "L": "香",
    "N": "桂",
    "S": "銀",
    "G": "金",
    "B": "角",
    "R": "飛",
    "K": "王",
    "+P": "と",
    "+L": "成香",
    "+N": "成桂",
    "+S": "成銀",
    "+B": "馬",
    "+R": "龍",
}


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
    def __init__(
        self,
        board: list[list[Optional[dict[str, Any]]]],
        turn: str,
        player_hand: Optional[list[str]] = None,
        enemy_hand: Optional[list[str]] = None,
    ):
        self.board = board
        self.turn = turn
        self.player_hand = player_hand or []
        self.enemy_hand = enemy_hand or []

    @classmethod
    def from_html_state(cls, data: dict[str, Any]) -> "ShogiBoard":
        converted_board = []

        for row in data["board"]:
            new_row = []

            for cell in row:
                if cell is None:
                    new_row.append(None)
                    continue

                new_row.append({
                    "type": PIECE_MAP.get(cell["type"], cell["type"]),
                    "owner": cell["owner"],
                })

            converted_board.append(new_row)

        return cls(
            board=converted_board,
            turn=data.get("turn", "enemy"),
            player_hand=cls.convert_hand(data.get("playerHand", [])),
            enemy_hand=cls.convert_hand(data.get("enemyHand", [])),
        )

    @staticmethod
    def convert_hand(hand: Any) -> list[str]:
        converted: list[str] = []

        if isinstance(hand, list):
            for piece in hand:
                converted.append(PIECE_MAP.get(piece, piece))
            return converted

        if isinstance(hand, dict):
            for piece, count in hand.items():
                for _ in range(int(count)):
                    converted.append(PIECE_MAP.get(piece, piece))
            return converted

        return converted

    def clone(self) -> "ShogiBoard":
        new_board = []

        for row in self.board:
            new_row = []

            for piece in row:
                if piece is None:
                    new_row.append(None)
                else:
                    new_row.append({
                        "type": piece["type"],
                        "owner": piece["owner"],
                    })

            new_board.append(new_row)

        return ShogiBoard(
            board=new_board,
            turn=self.turn,
            player_hand=self.player_hand.copy(),
            enemy_hand=self.enemy_hand.copy(),
        )

    def inside(self, row: int, col: int) -> bool:
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def owner_at(self, row: int, col: int) -> Optional[str]:
        piece = self.board[row][col]
        return None if piece is None else piece["owner"]

    def piece_at(self, row: int, col: int) -> Optional[dict[str, Any]]:
        return self.board[row][col]

    def enemy_of(self, owner: str) -> str:
        return "enemy" if owner == "player" else "player"

    def hand_of(self, owner: str) -> list[str]:
        return self.player_hand if owner == "player" else self.enemy_hand

    def forward(self, owner: str) -> int:
        return -1 if owner == "player" else 1

    def can_promote_zone(self, owner: str, row: int) -> bool:
        if owner == "player":
            return row <= 2
        return row >= 6

    def should_offer_promotion(self, owner: str, piece: str, from_row: int, to_row: int) -> bool:
        if piece not in PROMOTE_MAP:
            return False
        return self.can_promote_zone(owner, from_row) or self.can_promote_zone(owner, to_row)

    def should_force_promotion_choice(self, owner: str, piece: str, from_row: int, to_row: int) -> bool:
        if piece not in PROMOTE_MAP:
            return False

        if not self.should_offer_promotion(owner, piece, from_row, to_row):
            return False

        return True
    
    def must_promote(self, owner: str, piece: str, to_row: int) -> bool:
        if piece in ("P", "L"):
            return to_row == (0 if owner == "player" else 8)
        if piece == "N":
            return to_row in ((0, 1) if owner == "player" else (7, 8))
        return False

    def has_unpromoted_pawn_in_column(self, owner: str, col: int) -> bool:
        for row in range(BOARD_SIZE):
            piece = self.board[row][col]
            if piece and piece["owner"] == owner and piece["type"] == "P":
                return True
        return False

    def can_drop_piece(self, owner: str, piece: str, row: int, col: int) -> bool:
        if self.board[row][col] is not None:
            return False

        if piece in ("P", "L"):
            if row == (0 if owner == "player" else 8):
                return False

        if piece == "N":
            if owner == "player" and row in (0, 1):
                return False
            if owner == "enemy" and row in (7, 8):
                return False

        if piece == "P" and self.has_unpromoted_pawn_in_column(owner, col):
            return False

        return True

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

            if self.must_promote(owner, piece, nr):
                moves.append(Move(row, col, nr, nc, piece, promote=True))
                continue

            if self.should_force_promotion_choice(owner, piece, row, nr):
                moves.append(Move(row, col, nr, nc, piece, promote=True))
                continue

            moves.append(Move(row, col, nr, nc, piece, promote=False))

        for dr, dc in self.sliding_dirs(piece, owner):
            nr, nc = row + dr, col + dc

            while self.inside(nr, nc):
                target_owner = self.owner_at(nr, nc)

                if target_owner == owner:
                    break

                if self.must_promote(owner, piece, nr):
                    moves.append(Move(row, col, nr, nc, piece, promote=True))
                elif self.should_force_promotion_choice(owner, piece, row, nr):
                    moves.append(Move(row, col, nr, nc, piece, promote=True))
                else:
                    moves.append(Move(row, col, nr, nc, piece, promote=False))

                if target_owner == self.enemy_of(owner):
                    break

                nr += dr
                nc += dc

        return moves

    def generate_board_moves(self, owner: str) -> list[Move]:
        moves: list[Move] = []

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = self.piece_at(r, c)
                if piece and piece["owner"] == owner:
                    moves.extend(self.pseudo_moves_from_square(r, c))

        return moves

    def generate_drop_moves(self, owner: str) -> list[Move]:
        moves: list[Move] = []
        hand = self.hand_of(owner)

        for piece in sorted(set(hand)):
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if self.can_drop_piece(owner, piece, r, c):
                        moves.append(Move(None, None, r, c, piece, drop=True))

        return moves

    def generate_pseudo_moves(self, owner: str, include_drops: bool = False) -> list[Move]:
        moves = self.generate_board_moves(owner)
        if include_drops:
            moves.extend(self.generate_drop_moves(owner))
        return moves

    def find_king(self, owner: str) -> Optional[tuple[int, int]]:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = self.piece_at(r, c)
                if piece and piece["owner"] == owner and piece["type"] == "K":
                    return r, c
        return None

    def is_square_attacked(self, row: int, col: int, by_owner: str) -> bool:
        for move in self.generate_pseudo_moves(by_owner, include_drops=False):
            if move.to_row == row and move.to_col == col:
                return True
        return False

    def is_in_check(self, owner: str) -> bool:
        king_pos = self.find_king(owner)
        if king_pos is None:
            return True

        kr, kc = king_pos
        return self.is_square_attacked(kr, kc, self.enemy_of(owner))

    def normalize_captured_piece(self, piece_type: str) -> str:
        return UNPROMOTE_MAP.get(piece_type, piece_type)

    def apply_move(self, move: Move) -> None:
        owner = self.turn
        hand = self.hand_of(owner)

        if move.drop:
            if move.piece not in hand:
                raise ValueError(f"持ち駒に {move.piece} がありません")

            if not self.can_drop_piece(owner, move.piece, move.to_row, move.to_col):
                raise ValueError("その場所には駒を打てません")

            hand.remove(move.piece)
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

            captured = self.board[move.to_row][move.to_col]
            if captured is not None:
                hand.append(self.normalize_captured_piece(captured["type"]))

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

        for move in self.generate_pseudo_moves(owner, include_drops=True):
            copied = self.clone()
            copied.turn = owner
            copied.apply_move(move)

            if not copied.is_in_check(owner):
                legal.append(move)

        return legal