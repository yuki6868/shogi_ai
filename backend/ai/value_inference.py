# backend/ai/value_inference.py

from __future__ import annotations

import math
from pathlib import Path

import torch

from ai.board import ShogiBoard
from ai.board_tensor import board_to_full_tensor
from ai.path_config import WORKSPACE_DIR
from ai.value_model import create_value_model


DEFAULT_VALUE_MODEL_PATH = WORKSPACE_DIR / "models" / "value_model.pt"


class ValueInference:
    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_VALUE_MODEL_PATH
        self.device = self._get_device()
        self.model = None
        self.available = False

        self._load()

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    def _load(self) -> None:
        if not self.model_path.exists():
            print(f"[VALUE] model not found: {self.model_path}")
            return

        try:
            model = create_value_model(device=self.device)
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()

            self.model = model
            self.available = True

            print(f"[VALUE] loaded: {self.model_path} / device={self.device}")

        except Exception as e:
            print(f"[VALUE] load failed: {e}")
            self.model = None
            self.available = False

    @torch.no_grad()
    def predict_enemy_win_rate(self, shogi: ShogiBoard) -> float:
        """
        後手(enemy)の勝率を 0.0〜100.0 で返す。
        """
        if not self.available or self.model is None:
            raise RuntimeError("value_model.pt が読み込まれていません。")

        x = board_to_full_tensor(shogi).unsqueeze(0).to(self.device)

        logits = self.model(x)
        prob = torch.sigmoid(logits)[0].item()

        return round(prob * 100, 1)

    @torch.no_grad()
    def evaluate_score(self, shogi: ShogiBoard, ai_owner: str = "enemy") -> int:
        """
        既存 evaluator.py と合わせるために、
        勝率を評価値っぽいスコアへ変換する。

        + なら ai_owner 有利
        - なら ai_owner 不利
        """
        enemy_win_rate = self.predict_enemy_win_rate(shogi)
        p_enemy = enemy_win_rate / 100

        p_enemy = min(max(p_enemy, 0.001), 0.999)

        score = int(600 * math.log(p_enemy / (1 - p_enemy)))

        if ai_owner == "enemy":
            return score

        return -score


_value_inference: ValueInference | None = None


def get_value_inference() -> ValueInference:
    global _value_inference

    if _value_inference is None:
        _value_inference = ValueInference()

    return _value_inference