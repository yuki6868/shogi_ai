# backend/ai/value_inference.py

from __future__ import annotations

from pathlib import Path

import torch

from ai.board_tensor import board_to_full_tensor
from ai.path_config import WORKSPACE_DIR
from ai.value_model import create_value_model


DEFAULT_VALUE_MODEL_PATH = WORKSPACE_DIR / "models" / "value_model.pt"


class ValueInference:
    """
    value_model.pt を使って局面評価を行うクラス。

    ValueModel の出力は logits。
    sigmoid(logits) で enemy の勝率 0.0〜1.0 に変換する。
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_VALUE_MODEL_PATH
        self.device = torch.device(device or self._get_device())
        self.model = None
        self.available = False

        self._load_model()

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def _load_model(self) -> None:
        if not self.model_path.exists():
            print(f"[ValueInference] model not found: {self.model_path}")
            self.available = False
            return

        try:
            model = create_value_model(device=self.device)

            state_dict = torch.load(
                self.model_path,
                map_location=self.device,
            )

            model.load_state_dict(state_dict)
            model.eval()

            self.model = model
            self.available = True

            print(f"[ValueInference] loaded: {self.model_path}")
            print(f"[ValueInference] device: {self.device}")

        except Exception as e:
            print("[ValueInference] load failed:", e)
            self.model = None
            self.available = False

    @torch.no_grad()
    def predict_enemy_win_rate(self, shogi) -> float:
        """
        enemy の勝率を返す。
        return: 0.0〜1.0
        """

        if not self.available or self.model is None:
            return 0.5

        x = board_to_full_tensor(shogi)
        x = x.unsqueeze(0).to(self.device)

        logits = self.model(x)
        prob = torch.sigmoid(logits)

        value = float(prob.squeeze().item())
        return max(0.0, min(1.0, value))

    @torch.no_grad()
    def predict_value_for_owner(self, shogi, owner: str = "enemy") -> float:
        """
        owner から見た評価値を返す。
        return: -1.0〜1.0
        """

        enemy_win_rate = self.predict_enemy_win_rate(shogi)

        if owner == "enemy":
            owner_win_rate = enemy_win_rate
        else:
            owner_win_rate = 1.0 - enemy_win_rate

        return owner_win_rate * 2.0 - 1.0

    def evaluate_score(
        self,
        shogi,
        ai_owner: str = "enemy",
    ) -> int:
        """
        evaluator.py 互換の評価値を返す。
        return: おおよそ -1000〜+1000
        """

        value = self.predict_value_for_owner(
            shogi,
            owner=ai_owner,
        )

        return int(value * 1000)


_value_inference: ValueInference | None = None


def get_value_inference() -> ValueInference:
    global _value_inference

    if _value_inference is None:
        _value_inference = ValueInference()

    return _value_inference