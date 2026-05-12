# backend/ai/policy_inference.py

from __future__ import annotations

from pathlib import Path

import torch

from ai.board import Move, ShogiBoard
from ai.board_tensor import board_to_full_tensor
from ai.move_encoder import move_to_id
from ai.path_config import WORKSPACE_DIR
from ai.policy_dataset import (
    label_to_move_id,
    policy_output_size,
    move_id_to_label,
)
from ai.policy_model import create_policy_model


DEFAULT_POLICY_MODEL_PATH = WORKSPACE_DIR / "models" / "policy_model.pt"


class PolicyInference:
    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_POLICY_MODEL_PATH
        self.device = torch.device(device or self._get_device())
        self.model = None
        self.available = False

        self._load_model()

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def _load_model(self) -> None:
        if not self.model_path.exists():
            print(f"[PolicyInference] model not found: {self.model_path}")
            self.available = False
            return

        model = create_policy_model(device=self.device)

        state_dict = torch.load(
            self.model_path,
            map_location=self.device,
        )

        model.load_state_dict(state_dict)
        model.eval()

        self.model = model
        self.available = True

        print(f"[PolicyInference] loaded: {self.model_path}")
        print(f"[PolicyInference] device: {self.device}")
        print(f"[PolicyInference] output size: {policy_output_size()}")

    @torch.no_grad()
    def rank_legal_moves(
        self,
        shogi: ShogiBoard,
        legal_moves: list[Move],
        top_k: int = 12,
    ) -> list[dict]:
        if not legal_moves:
            return []

        if not self.available or self.model is None:
            return []

        x = board_to_full_tensor(shogi)
        x = x.unsqueeze(0).to(self.device)

        logits = self.model(x)[0]

        ranked: list[dict] = []

        for move in legal_moves:
            move_id = move_to_id(move)

            try:
                label = self._move_id_to_label_safe(move_id)
            except Exception:
                continue

            score = float(logits[label].item())

            ranked.append(
                {
                    "move": move,
                    "moveId": move_id,
                    "policyScore": score,
                }
            )

        ranked.sort(key=lambda item: item["policyScore"], reverse=True)

        return ranked[:top_k]

    def _move_id_to_label_safe(self, move_id: str) -> int:

        return move_id_to_label(move_id)


_policy_inference: PolicyInference | None = None


def get_policy_inference() -> PolicyInference:
    global _policy_inference

    if _policy_inference is None:
        _policy_inference = PolicyInference()

    return _policy_inference


def filter_policy_ai_candidates(
    shogi: ShogiBoard,
    legal_moves: list[Move],
    top_k: int = 12,
) -> list[Move]:
    policy = get_policy_inference()

    ranked = policy.rank_legal_moves(
        shogi=shogi,
        legal_moves=legal_moves,
        top_k=top_k,
    )

    if not ranked:
        return legal_moves[:top_k]

    return [item["move"] for item in ranked]


def policy_ai_candidates_to_dicts(
    shogi: ShogiBoard,
    legal_moves: list[Move],
    limit: int = 12,
) -> list[dict]:
    from ai.move_encoder import move_to_readable

    policy = get_policy_inference()

    ranked = policy.rank_legal_moves(
        shogi=shogi,
        legal_moves=legal_moves,
        top_k=limit,
    )

    return [
        {
            "moveId": item["moveId"],
            "moveText": move_to_readable(item["move"]),
            "policyScore": round(item["policyScore"], 4),
            "move": item["move"].to_dict(),
        }
        for item in ranked
    ]