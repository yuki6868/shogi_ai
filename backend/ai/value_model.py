# backend/ai/value_model.py

from __future__ import annotations

import torch
from torch import nn


INPUT_CHANNELS = 44


class ValueModel(nn.Module):
    """
    局面から「後手(enemy)が最終的に勝つ確率」を予測するAI。
    出力:
        shape = (batch,)
        値域 = 0.0 〜 1.0
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 9 * 9, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.head(x)
        return logits.squeeze(1)


def create_value_model(device: str | torch.device | None = None) -> ValueModel:
    model = ValueModel()

    if device is not None:
        model = model.to(device)

    return model