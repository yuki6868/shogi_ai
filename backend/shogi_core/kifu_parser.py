from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from shogi_core.board import BOARD_SIZE, ShogiBoard, Move
from shogi_core.move_encoder import move_to_id, find_legal_move_by_id


CSA_TO_PIECE = {
    "FU": "P",
    "KY": "L",
    "KE": "N",
    "GI": "S",
    "KI": "G",
    "KA": "B",
    "HI": "R",
    "OU": "K",
    "TO": "+P",
    "NY": "+L",
    "NK": "+N",
    "NG": "+S",
    "UM": "+B",
    "RY": "+R",
}

CSA_BLACK = "+"
CSA_WHITE = "-"

CSA_SIDE_TO_OWNER = {
    CSA_BLACK: "player",
    CSA_WHITE: "enemy",
}


@dataclass(frozen=True)
class ParsedMoveRecord:
    ply: int
    owner: str
    move_id: str
    move: Move
    board_before: ShogiBoard
    csa_line: str


class KifuParseError(ValueError):
    pass


def create_empty_board() -> ShogiBoard:
    return ShogiBoard(
        board=[[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)],
        turn="player",
        player_hand=[],
        enemy_hand=[],
    )


def create_initial_board() -> ShogiBoard:
    board = create_empty_board()

    enemy_back = ["L", "N", "S", "G", "K", "G", "S", "N", "L"]
    player_back = ["L", "N", "S", "G", "K", "G", "S", "N", "L"]

    for col, piece in enumerate(enemy_back):
        board.board[0][col] = {"type": piece, "owner": "enemy"}

    board.board[1][1] = {"type": "R", "owner": "enemy"}
    board.board[1][7] = {"type": "B", "owner": "enemy"}

    for col in range(BOARD_SIZE):
        board.board[2][col] = {"type": "P", "owner": "enemy"}

    for col in range(BOARD_SIZE):
        board.board[6][col] = {"type": "P", "owner": "player"}

    board.board[7][1] = {"type": "B", "owner": "player"}
    board.board[7][7] = {"type": "R", "owner": "player"}

    for col, piece in enumerate(player_back):
        board.board[8][col] = {"type": piece, "owner": "player"}

    return board


def parse_position_lines(position_lines: list[str]) -> ShogiBoard:
    if any(line.strip() == "PI" for line in position_lines):
        return create_initial_board()

    board = create_empty_board()

    for line in position_lines:
        line = line.rstrip("\n")

        if len(line) < 2:
            continue

        # P1〜P9: 盤面
        if line[0] == "P" and line[1] in "123456789":
            row = int(line[1]) - 1
            body = line[2:]

            if len(body) < 27:
                raise KifuParseError(f"盤面行が短すぎます: {line}")

            for col in range(9):
                cell = body[col * 3: col * 3 + 3]

                if cell == " * ":
                    board.board[row][col] = None
                    continue

                side = cell[0]
                csa_piece = cell[1:3]

                if side not in (CSA_BLACK, CSA_WHITE):
                    raise KifuParseError(f"駒の所有者が不正です: {cell} / {line}")

                if csa_piece not in CSA_TO_PIECE:
                    raise KifuParseError(f"未対応の駒です: {cell} / {line}")

                board.board[row][col] = {
                    "type": CSA_TO_PIECE[csa_piece],
                    "owner": CSA_SIDE_TO_OWNER[side],
                }

        # P+ / P-: 持ち駒
        elif line.startswith("P+") or line.startswith("P-"):
            owner = CSA_SIDE_TO_OWNER[line[1]]
            hand = board.hand_of(owner)
            body = line[2:]

            for i in range(0, len(body), 4):
                chunk = body[i:i + 4]
                if len(chunk) < 4:
                    continue

                # 例: 00FU
                if chunk[:2] != "00":
                    continue

                csa_piece = chunk[2:4]
                if csa_piece in CSA_TO_PIECE:
                    hand.append(CSA_TO_PIECE[csa_piece].lstrip("+"))

    return board


def csa_square_to_row_col(file_char: str, rank_char: str) -> tuple[int, int]:
    if file_char not in "123456789" or rank_char not in "123456789":
        raise KifuParseError(f"CSA座標が不正です: {file_char}{rank_char}")

    file_no = int(file_char)
    rank_no = int(rank_char)

    row = rank_no - 1
    col = 9 - file_no
    return row, col


def parse_csa_move_line(line: str, board: ShogiBoard) -> tuple[str, Move]:
    line = line.strip()

    if len(line) != 7 or line[0] not in (CSA_BLACK, CSA_WHITE):
        raise KifuParseError(f"CSA指し手行が不正です: {line}")

    side = line[0]
    owner = CSA_SIDE_TO_OWNER[side]

    from_file = line[1]
    from_rank = line[2]
    to_file = line[3]
    to_rank = line[4]
    csa_piece = line[5:7]

    if csa_piece not in CSA_TO_PIECE:
        raise KifuParseError(f"未対応のCSA駒種です: {csa_piece} / line={line}")

    piece_after = CSA_TO_PIECE[csa_piece]
    to_row, to_col = csa_square_to_row_col(to_file, to_rank)

    if from_file == "0" and from_rank == "0":
        return owner, Move(
            from_row=None,
            from_col=None,
            to_row=to_row,
            to_col=to_col,
            piece=piece_after.lstrip("+"),
            promote=False,
            drop=True,
        )

    from_row, from_col = csa_square_to_row_col(from_file, from_rank)
    piece_data = board.piece_at(from_row, from_col)

    if piece_data is None:
        raise KifuParseError(f"移動元に駒がありません: {line}")

    if piece_data["owner"] != owner:
        raise KifuParseError(
            f"手番と移動元の駒の所有者が一致しません: "
            f"line={line}, owner={owner}, piece_owner={piece_data['owner']}"
        )

    piece_before = piece_data["type"]
    promote = piece_before != piece_after

    return owner, Move(
        from_row=from_row,
        from_col=from_col,
        to_row=to_row,
        to_col=to_col,
        piece=piece_before,
        promote=promote,
        drop=False,
    )


def is_csa_move_line(line: str) -> bool:
    line = line.strip()
    return len(line) == 7 and line[0] in (CSA_BLACK, CSA_WHITE) and line[1:5].isdigit()


def parse_csa_text(text: str, strict_legal: bool = True) -> list[ParsedMoveRecord]:
    lines = [line.strip("\n") for line in text.splitlines()]

    position_lines: list[str] = []
    move_start_index = 0

    for i, raw_line in enumerate(lines):
        line = raw_line.strip()

        if line == "+" or line == "-":
            move_start_index = i + 1
            break

        if line == "PI" or line.startswith("P"):
            position_lines.append(raw_line)

    if not position_lines:
        raise KifuParseError("初期局面が見つかりません。PI または P1〜P9 が必要です。")

    board = parse_position_lines(position_lines)

    if move_start_index > 0:
        board.turn = CSA_SIDE_TO_OWNER[lines[move_start_index - 1].strip()]

    records: list[ParsedMoveRecord] = []
    ply = 0

    for raw_line in lines[move_start_index:]:
        line = raw_line.strip()

        if not line:
            continue
        if line.startswith("'"):
            continue
        if line.startswith(("T", "$", "V", "N+", "N-")):
            continue
        if line.startswith("%"):
            break
        if not is_csa_move_line(line):
            continue

        owner, parsed_move = parse_csa_move_line(line, board)
        board.turn = owner

        board_before = board.clone()
        parsed_move_id = move_to_id(parsed_move)

        if strict_legal:
            legal_moves = board.generate_legal_moves(owner)
            legal_move = find_legal_move_by_id(parsed_move_id, legal_moves)
        else:
            legal_move = parsed_move

        ply += 1

        records.append(
            ParsedMoveRecord(
                ply=ply,
                owner=owner,
                move_id=move_to_id(legal_move),
                move=legal_move,
                board_before=board_before,
                csa_line=line,
            )
        )

        board.apply_move(legal_move)

    return records


def parse_csa_file(path: str | Path, strict_legal: bool = True) -> list[ParsedMoveRecord]:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    return parse_csa_text(text, strict_legal=strict_legal)


def parse_csa_files(
    paths: Iterable[str | Path],
    strict_legal: bool = True,
    skip_errors: bool = True,
) -> list[ParsedMoveRecord]:
    records: list[ParsedMoveRecord] = []

    for path in paths:
        try:
            records.extend(parse_csa_file(path, strict_legal=strict_legal))
        except Exception as e:
            if skip_errors:
                print(f"[SKIP] {path}: {e}")
                continue
            raise

    return records


def iter_csa_files(dataset_dir: str | Path) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    return sorted(
        list(dataset_dir.glob("**/*.csa"))
        + list(dataset_dir.glob("**/*.CSA"))
    )


def records_to_training_rows(records: Iterable[ParsedMoveRecord]) -> list[dict[str, object]]:
    return [
        {
            "ply": record.ply,
            "owner": record.owner,
            "move_id": record.move_id,
            "csa_line": record.csa_line,
        }
        for record in records
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CSA棋譜をmove_id教師データに変換します")
    parser.add_argument("path", help="CSAファイル、またはCSAファイルを含むディレクトリ")
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    target = Path(args.path)
    strict = not args.no_strict

    if target.is_dir():
        files = iter_csa_files(target)
        if args.max_files > 0:
            files = files[:args.max_files]
        all_records = parse_csa_files(files, strict_legal=strict, skip_errors=True)
    else:
        all_records = parse_csa_file(target, strict_legal=strict)

    print(f"parsed_records={len(all_records)}")

    for row in records_to_training_rows(all_records[: args.limit]):
        print(row)