# backend/ai/value_dataset.py

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from shogi_core.board_tensor import board_to_full_tensor
from shogi_core.kifu_parser import (
    ParsedMoveRecord,
    iter_csa_files,
    parse_csa_file,
)
from shogi_core.path_config import DEFAULT_DATASET_DIR


def infer_winner_from_records(records: list[ParsedMoveRecord]) -> str | None:
    """
    CSAの %TORYO は、基本的に「次に指す側が投了」。
    つまり最後に指した側が勝者。
    """
    if not records:
        return None

    last_owner = records[-1].owner

    if last_owner == "player":
        return "player"

    if last_owner == "enemy":
        return "enemy"

    return None


class ValueDataset(Dataset):
    """
    局面 → 最終的に後手(enemy)が勝ったか

    y:
        enemy勝ち  = 1.0
        player勝ち = 0.0
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

        self.samples: list[tuple[ParsedMoveRecord, float]] = []

        for path in files:
            try:
                records = parse_csa_file(path, strict_legal=strict_legal)
                winner = infer_winner_from_records(records)

                if winner is None:
                    continue

                label = 1.0 if winner == "enemy" else 0.0

                for record in records:
                    self.samples.append((record, label))

            except Exception as e:
                if skip_errors:
                    print(f"[SKIP] {path}: {e}")
                    continue
                raise

        print(f"value dataset files = {len(files)}")
        print(f"value dataset samples = {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record, label = self.samples[index]

        x = board_to_full_tensor(record.board_before)
        y = torch.tensor(label, dtype=torch.float32)

        return x, y


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", nargs="?", default=None)
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()

    dataset = ValueDataset(
        dataset_dir=args.dataset_dir,
        max_files=args.max_files,
        strict_legal=not args.no_strict,
        skip_errors=True,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for x, y in loader:
        print("x shape =", x.shape)
        print("y shape =", y.shape)
        print("y =", y)
        break