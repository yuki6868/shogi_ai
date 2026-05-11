# backend/tools/build_processed_dataset.py

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import torch
from tqdm import tqdm

import sys

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent

sys.path.insert(0, str(BACKEND_DIR))

from ai.kifu_parser import iter_csa_files, parse_csa_file
from ai.board_tensor import board_to_full_tensor
from ai.move_encoder import move_to_id


# =========================================================
# 設定
# =========================================================

DATASET_DIR = BACKEND_DIR.parent.parent / "dataset"

OUTPUT_DIR = BACKEND_DIR.parent.parent / "processed_dataset"

MAX_FILES = 5000

STRICT_LEGAL = False

print("DATASET_DIR:", DATASET_DIR)
print("exists:", DATASET_DIR.exists())


# =========================================================
# 初期化
# =========================================================

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

policy_dir = OUTPUT_DIR / "policy"
value_dir = OUTPUT_DIR / "value"

policy_dir.mkdir(parents=True, exist_ok=True)
value_dir.mkdir(parents=True, exist_ok=True)

print("DATASET_DIR :", DATASET_DIR)
print("OUTPUT_DIR  :", OUTPUT_DIR)

files = list(iter_csa_files(DATASET_DIR))

if MAX_FILES > 0:
    files = files[:MAX_FILES]

print("files:", len(files))


# =========================================================
# move vocabulary
# =========================================================

move_vocab: dict[str, int] = {}
move_vocab_reverse: dict[int, str] = {}

next_move_class = 0


def encode_move(move_text: str) -> int:
    global next_move_class

    if move_text not in move_vocab:
        move_vocab[move_text] = next_move_class
        move_vocab_reverse[next_move_class] = move_text

        next_move_class += 1

    return move_vocab[move_text]


# =========================================================
# dataset build
# =========================================================

policy_count = 0
value_count = 0

skip_files = 0
skip_policy = 0
skip_value = 0

for file_index, path in enumerate(tqdm(files)):

    try:

        records = parse_csa_file(
            path,
            strict_legal=STRICT_LEGAL,
        )

        if not records:
            continue

        winner = records[-1].owner

        value_label = 1.0 if winner == "enemy" else 0.0

        for move_index, record in enumerate(records):

            x = board_to_full_tensor(
                record.board_before
            ).cpu()

            # =====================================
            # policy dataset
            # =====================================

            try:

                move_text = move_to_id(record.move)

                y_policy = encode_move(move_text)

                policy_path = (
                    policy_dir
                    / f"{policy_count:08d}.pt"
                )

                torch.save(
                    {
                        "x": x,
                        "y": y_policy,
                        "move_text": move_text,
                    },
                    policy_path,
                )

                policy_count += 1

            except Exception as e:

                skip_policy += 1

                print()
                print("policy skip")
                print(path)
                print(e)

            # =====================================
            # value dataset
            # =====================================

            try:

                value_path = (
                    value_dir
                    / f"{value_count:08d}.pt"
                )

                torch.save(
                    {
                        "x": x,
                        "y": float(value_label),
                    },
                    value_path,
                )

                value_count += 1

            except Exception as e:

                skip_value += 1

                print()
                print("value skip")
                print(path)
                print(e)

    except Exception as e:

        skip_files += 1

        print()
        print("file skip")
        print(path)
        print(e)


# =========================================================
# save vocab
# =========================================================

move_vocab_path = OUTPUT_DIR / "move_vocab.json"

with open(
    move_vocab_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        move_vocab,
        f,
        ensure_ascii=False,
        indent=2,
    )

move_vocab_reverse_path = (
    OUTPUT_DIR / "move_vocab_reverse.json"
)

with open(
    move_vocab_reverse_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            str(k): v
            for k, v in move_vocab_reverse.items()
        },
        f,
        ensure_ascii=False,
        indent=2,
    )


# =========================================================
# zip
# =========================================================

zip_path = BACKEND_DIR.parent / "processed_dataset.zip"

if zip_path.exists():
    zip_path.unlink()

print()
print("creating zip...")

with zipfile.ZipFile(
    zip_path,
    "w",
    compression=zipfile.ZIP_DEFLATED,
) as z:

    for path in OUTPUT_DIR.rglob("*"):

        if path.is_file():

            arcname = path.relative_to(OUTPUT_DIR)

            z.write(
                path,
                arcname=str(arcname),
            )


# =========================================================
# result
# =========================================================

print()
print("========================================")
print("DONE")
print("========================================")

print()

print("policy samples :", policy_count)
print("value samples  :", value_count)

print()

print("move classes   :", len(move_vocab))

print()

print("skip files     :", skip_files)
print("skip policy    :", skip_policy)
print("skip value     :", skip_value)

print()

print("saved dataset  :", OUTPUT_DIR)
print("saved zip      :", zip_path)

print()

print(
    "zip size MB    :",
    round(
        zip_path.stat().st_size / 1024 / 1024,
        2,
    ),
)