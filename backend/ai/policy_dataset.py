# backend/ai/policy_dataset.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from ai.board_tensor import board_to_full_tensor
from ai.kifu_parser import (
    ParsedMoveRecord,
    iter_csa_files,
    parse_csa_file,
)
from ai.move_encoder import (
    FILES,
    RANKS,
    DROP_PIECES,
)
from ai.path_config import DEFAULT_DATASET_DIR

MOVE_ID_TO_LABEL: dict[str, int] = {}
LABEL_TO_MOVE_ID: dict[int, str] = {}


def build_all_move_ids() -> list[str]:
    """
    AIの出力候補になる全move_idを作る。

    通常手:
        7g7f
        8h2b+

    持ち駒打ち:
        P*5e
    """

    move_ids: list[str] = []

    squares = [
        f"{file}{rank}"
        for file in FILES
        for rank in RANKS
    ]

    # 通常移動
    for from_sq in squares:
        for to_sq in squares:
            if from_sq == to_sq:
                continue

            move_ids.append(f"{from_sq}{to_sq}")
            move_ids.append(f"{from_sq}{to_sq}+")

    # 持ち駒打ち
    for piece in sorted(DROP_PIECES):
        for to_sq in squares:
            move_ids.append(f"{piece}*{to_sq}")

    return move_ids


def build_move_label_maps() -> None:
    global MOVE_ID_TO_LABEL
    global LABEL_TO_MOVE_ID

    if MOVE_ID_TO_LABEL and LABEL_TO_MOVE_ID:
        return

    all_move_ids = build_all_move_ids()

    MOVE_ID_TO_LABEL = {
        move_id: label
        for label, move_id in enumerate(all_move_ids)
    }

    LABEL_TO_MOVE_ID = {
        label: move_id
        for move_id, label in MOVE_ID_TO_LABEL.items()
    }


def move_id_to_label(move_id: str) -> int:
    build_move_label_maps()

    if move_id not in MOVE_ID_TO_LABEL:
        raise ValueError(f"未知のmove_idです: {move_id}")

    return MOVE_ID_TO_LABEL[move_id]


def label_to_move_id(label: int) -> str:
    build_move_label_maps()

    if label not in LABEL_TO_MOVE_ID:
        raise ValueError(f"未知のlabelです: {label}")

    return LABEL_TO_MOVE_ID[label]


def policy_output_size() -> int:
    build_move_label_maps()
    return len(MOVE_ID_TO_LABEL)


class PolicyDataset(Dataset):
    """
    CSA棋譜をPyTorch Datasetとして扱う。

    return:
        x: board tensor
        y: move label
    """

    def __init__(
        self,
        dataset_dir: str | Path | None = None,
        max_files: int = 0,
        strict_legal: bool = True,
        skip_errors: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir) if dataset_dir else DEFAULT_DATASET_DIR
        self.strict_legal = strict_legal
        self.skip_errors = skip_errors

        files = iter_csa_files(self.dataset_dir)

        if max_files > 0:
            files = files[:max_files]

        self.records: list[ParsedMoveRecord] = []

        for path in files:
            try:
                records = parse_csa_file(path, strict_legal=strict_legal)
                self.records.extend(records)
            except Exception as e:
                if skip_errors:
                    print(f"[SKIP] {path}: {e}")
                    continue
                raise

        build_move_label_maps()

        print(f"dataset files = {len(files)}")
        print(f"dataset records = {len(self.records)}")
        print(f"policy output size = {policy_output_size()}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]

        x = board_to_full_tensor(record.board_before)
        y = torch.tensor(
            move_id_to_label(record.move_id),
            dtype=torch.long,
        )

        return x, y


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=None,
        help="CSA棋譜フォルダ。省略時は shogi_ai と同じ階層の dataset/",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
    )

    args = parser.parse_args()

    dataset = PolicyDataset(
        dataset_dir=args.dataset_dir,
        max_files=args.max_files,
        strict_legal=not args.no_strict,
        skip_errors=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    for x, y in loader:
        print("x shape =", x.shape)
        print("y shape =", y.shape)
        print("y =", y)
        print("move_id =", label_to_move_id(int(y[0])))
        break