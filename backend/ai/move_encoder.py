# backend/ai/move_encoder.py

from __future__ import annotations

from typing import Iterable

from ai.board import BOARD_SIZE, Move


FILES = "987654321"
RANKS = "abcdefghi"
DROP_PIECES = {"P", "L", "N", "S", "G", "B", "R"}

JAPANESE_FILES = "９８７６５４３２１"
JAPANESE_RANKS = "一二三四五六七八九"

PIECE_LABELS = {
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


class MoveEncodeError(ValueError):
    pass


def square_to_id(row: int, col: int) -> str:
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise MoveEncodeError(f"盤外の座標です: row={row}, col={col}")

    return f"{FILES[col]}{RANKS[row]}"


def square_from_id(square_id: str) -> tuple[int, int]:
    if len(square_id) != 2:
        raise MoveEncodeError(f"マスIDは2文字である必要があります: {square_id}")

    file_char = square_id[0]
    rank_char = square_id[1]

    if file_char not in FILES:
        raise MoveEncodeError(f"筋が不正です: {file_char}")
    if rank_char not in RANKS:
        raise MoveEncodeError(f"段が不正です: {rank_char}")

    row = RANKS.index(rank_char)
    col = FILES.index(file_char)
    return row, col


def move_to_id(move: Move) -> str:
    to_sq = square_to_id(move.to_row, move.to_col)

    if move.drop:
        piece = normalize_piece_for_drop(move.piece)
        return f"{piece}*{to_sq}"

    if move.from_row is None or move.from_col is None:
        raise MoveEncodeError("通常移動には from_row/from_col が必要です")

    from_sq = square_to_id(move.from_row, move.from_col)
    suffix = "+" if move.promote else ""
    return f"{from_sq}{to_sq}{suffix}"


def move_from_id(move_id: str, piece: str = "") -> Move:
    if is_drop_move_id(move_id):
        return drop_move_from_id(move_id)

    promote = move_id.endswith("+")
    body = move_id[:-1] if promote else move_id

    if len(body) != 4:
        raise MoveEncodeError(f"通常移動のmove_idが不正です: {move_id}")

    if not piece:
        raise MoveEncodeError(
            "通常移動をMoveに戻すには piece が必要です。"
            "合法手から戻す場合は find_legal_move_by_id() を使ってください。"
        )

    from_row, from_col = square_from_id(body[:2])
    to_row, to_col = square_from_id(body[2:4])

    return Move(
        from_row=from_row,
        from_col=from_col,
        to_row=to_row,
        to_col=to_col,
        piece=piece,
        promote=promote,
        drop=False,
    )


def drop_move_from_id(move_id: str) -> Move:
    if len(move_id) != 4 or move_id[1] != "*":
        raise MoveEncodeError(f"持ち駒打ちのmove_idが不正です: {move_id}")

    piece = normalize_piece_for_drop(move_id[0])
    to_row, to_col = square_from_id(move_id[2:4])

    return Move(
        from_row=None,
        from_col=None,
        to_row=to_row,
        to_col=to_col,
        piece=piece,
        promote=False,
        drop=True,
    )


def is_drop_move_id(move_id: str) -> bool:
    return len(move_id) == 4 and move_id[1] == "*"


def normalize_piece_for_drop(piece: str) -> str:
    unpromote = {
        "+P": "P",
        "+L": "L",
        "+N": "N",
        "+S": "S",
        "+B": "B",
        "+R": "R",
    }

    normalized = unpromote.get(piece, piece)

    if normalized not in DROP_PIECES:
        raise MoveEncodeError(f"持ち駒打ちに使えない駒です: {piece}")

    return normalized


def legal_moves_to_ids(legal_moves: Iterable[Move]) -> list[str]:
    return [move_to_id(move) for move in legal_moves]


def legal_move_id_set(legal_moves: Iterable[Move]) -> set[str]:
    return set(legal_moves_to_ids(legal_moves))


def filter_legal_move_ids(
    candidate_move_ids: Iterable[str],
    legal_moves: Iterable[Move],
) -> list[str]:
    legal_ids = legal_move_id_set(legal_moves)
    return [move_id for move_id in candidate_move_ids if move_id in legal_ids]


def find_legal_move_by_id(move_id: str, legal_moves: Iterable[Move]) -> Move:
    for move in legal_moves:
        if move_to_id(move) == move_id:
            return move

    raise MoveEncodeError(f"合法手の中に存在しないmove_idです: {move_id}")


def build_legal_move_map(legal_moves: Iterable[Move]) -> dict[str, Move]:
    return {
        move_to_id(move): move
        for move in legal_moves
    }


def select_best_legal_policy_move(
    policy_scores: dict[str, float],
    legal_moves: Iterable[Move],
) -> Move:
    legal_map = build_legal_move_map(legal_moves)

    best_move_id: str | None = None
    best_score = float("-inf")

    for move_id, move in legal_map.items():
        score = policy_scores.get(move_id, float("-inf"))

        if score > best_score:
            best_score = score
            best_move_id = move_id

    if best_move_id is None:
        raise MoveEncodeError("合法手がありません")

    return legal_map[best_move_id]

def move_to_dict_with_id(move: Move) -> dict:
    data = move.to_dict()
    data["moveId"] = move_to_id(move)
    return data

JAPANESE_FILES = "９８７６５４３２１"
JAPANESE_RANKS = "一二三四五六七八九"

PIECE_LABELS = {
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


def square_to_japanese(row: int, col: int) -> str:
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise MoveEncodeError(f"盤外の座標です: row={row}, col={col}")

    return f"{JAPANESE_FILES[col]}{JAPANESE_RANKS[row]}"


def move_to_readable(move: Move) -> str:
    piece_name = PIECE_LABELS.get(move.piece, move.piece)
    to_sq = square_to_japanese(move.to_row, move.to_col)

    if move.drop:
        return f"{piece_name}を{to_sq}に打つ"

    if move.from_row is None or move.from_col is None:
        return f"{piece_name}を{to_sq}へ"

    from_sq = square_to_japanese(move.from_row, move.from_col)
    promote_text = " 成る" if move.promote else ""

    return f"{piece_name}：{from_sq} → {to_sq}{promote_text}"

def square_to_japanese(row: int, col: int) -> str:
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise MoveEncodeError(f"盤外の座標です: row={row}, col={col}")

    return f"{JAPANESE_FILES[col]}{JAPANESE_RANKS[row]}"


def move_to_readable(move: Move) -> str:
    piece_name = PIECE_LABELS.get(move.piece, move.piece)
    to_sq = square_to_japanese(move.to_row, move.to_col)

    if move.drop:
        return f"{piece_name}を{to_sq}に打つ"

    if move.from_row is None or move.from_col is None:
        return f"{piece_name}を{to_sq}へ"

    from_sq = square_to_japanese(move.from_row, move.from_col)
    promote_text = "・成る" if move.promote else ""

    return f"{piece_name}：{from_sq} → {to_sq}{promote_text}"


def move_to_dict_with_id(move: Move) -> dict:
    data = move.to_dict()
    data["moveId"] = move_to_id(move)
    data["moveText"] = move_to_readable(move)
    return data