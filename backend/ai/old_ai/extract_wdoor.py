# backend/ai/extract_wdoor.py

from __future__ import annotations

from pathlib import Path
import argparse

import py7zr

from shogi_core.path_config import DEFAULT_DATASET_DIR, find_wdoor_archive


def extract_wdoor(
    archive_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> None:
    archive = Path(archive_path) if archive_path else find_wdoor_archive()
    out = Path(output_dir) if output_dir else DEFAULT_DATASET_DIR

    out.mkdir(parents=True, exist_ok=True)

    print(f"archive = {archive}")
    print(f"output  = {out}")

    with py7zr.SevenZipFile(archive, mode="r") as z:
        z.extractall(path=out)

    print("展開完了")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_path", nargs="?", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    extract_wdoor(args.archive_path, args.output)