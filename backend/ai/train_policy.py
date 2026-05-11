# backend/ai/train_policy.py

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from ai.policy_dataset import PolicyDataset, policy_output_size
from ai.policy_model import create_policy_model
from ai.path_config import DEFAULT_DATASET_DIR, WORKSPACE_DIR

DEFAULT_MODEL_DIR = WORKSPACE_DIR / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "policy_model.pt"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)

        pred = torch.argmax(logits, dim=1)
        total_correct += int((pred == y).sum().item())
        total_count += int(x.size(0))

    avg_loss = total_loss / max(total_count, 1)
    accuracy = total_correct / max(total_count, 1)

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * x.size(0)

        pred = torch.argmax(logits, dim=1)
        total_correct += int((pred == y).sum().item())
        total_count += int(x.size(0))

    avg_loss = total_loss / max(total_count, 1)
    accuracy = total_correct / max(total_count, 1)

    return avg_loss, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="候補手AI PolicyModelを学習します")

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help="CSA棋譜フォルダ。省略時は shogi_ai と同じ階層の dataset/",
    )
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument("--save-path", type=str, default=str(DEFAULT_MODEL_PATH))

    args = parser.parse_args()

    device = get_device()
    print(f"device = {device}")

    dataset = PolicyDataset(
        dataset_dir=args.dataset_dir,
        max_files=args.max_files,
        strict_legal=not args.no_strict,
        skip_errors=True,
    )

    if len(dataset) == 0:
        raise RuntimeError("学習データが0件です。datasetフォルダにCSAファイルがあるか確認してください。")

    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size

    if val_size == 0:
        train_dataset = dataset
        val_dataset = None
    else:
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

    print(f"train size = {len(train_dataset)}")
    print(f"val size = {0 if val_dataset is None else len(val_dataset)}")
    print(f"policy output size = {policy_output_size()}")

    model = create_policy_model(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        print(
            f"[epoch {epoch}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f}"
        )

        if val_loader is not None:
            val_loss, val_acc = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            print(
                f"[epoch {epoch}] "
                f"val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print(f"saved best model: {save_path}")
        else:
            torch.save(model.state_dict(), save_path)
            print(f"saved model: {save_path}")

    print("training finished")


if __name__ == "__main__":
    main()