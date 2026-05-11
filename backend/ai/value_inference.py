# backend/ai/value_inference.py

from __future__ import annotations

import math
from pathlib import Path

import torch

from ai.board import ShogiBoard
from ai.board_tensor import board_to_full_tensor
from ai.kifu_parser import create_initial_board
from ai.path_config import WORKSPACE_DIR
from ai.value_model import create_value_model


DEFAULT_VALUE_MODEL_PATH = WORKSPACE_DIR / "models" / "value_model.pt"


class ValueInference:
    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_VALUE_MODEL_PATH
        self.device = self._get_device()
        self.model = None
        self.available = False

        # 初期局面の偏り補正
        self.initial_logit_bias = 0.0

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

            self.initial_logit_bias = self._calculate_initial_logit_bias()

            print(f"[VALUE] loaded: {self.model_path} / device={self.device}")
            print(f"[VALUE] initial_logit_bias = {self.initial_logit_bias:.4f}")

        except Exception as e:
            print(f"[VALUE] load failed: {e}")
            self.model = None
            self.available = False
            self.initial_logit_bias = 0.0

    @torch.no_grad()
    def _predict_raw_logit(self, shogi: ShogiBoard) -> float:
        if not self.available or self.model is None:
            raise RuntimeError("value_model.pt が読み込まれていません。")

        x = board_to_full_tensor(shogi).unsqueeze(0).to(self.device)
        logits = self.model(x)

        return float(logits[0].item())

    @torch.no_grad()
    def _calculate_initial_logit_bias(self) -> float:
        """
        初期局面は本来ほぼ互角なので、
        value_model.pt が初期局面に出す偏りを補正値として保存する。
        """
        if self.model is None:
            return 0.0

        initial_board = create_initial_board()
        initial_board.turn = "player"

        x = board_to_full_tensor(initial_board).unsqueeze(0).to(self.device)
        logits = self.model(x)

        return float(logits[0].item())

    def predict_enemy_win_rate(self, shogi: ShogiBoard) -> float:
        """
        後手(enemy)の勝率を 0.0〜100.0 で返す。
        初期局面の偏りを補正する。
        """
        raw_logit = self._predict_raw_logit(shogi)

        # ここが重要：初期局面の偏りを引く
        calibrated_logit = raw_logit - self.initial_logit_bias

        prob = 1.0 / (1.0 + math.exp(-calibrated_logit))

        return round(prob * 100, 1)

    def evaluate_score(self, shogi: ShogiBoard, ai_owner: str = "enemy") -> int:
        """
        + なら ai_owner 有利
        - なら ai_owner 不利
        """
        raw_logit = self._predict_raw_logit(shogi)

        # 初期局面が0になるように補正
        calibrated_logit = raw_logit - self.initial_logit_bias

        # logitをそのまま評価値スケールへ変換
        score = int(600 * calibrated_logit)

        if ai_owner == "enemy":
            return score

        return -score


_value_inference: ValueInference | None = None


def get_value_inference() -> ValueInference:
    global _value_inference

    if _value_inference is None:
        _value_inference = ValueInference()

    return _value_inference