# backend/ai/policy_model.py

from __future__ import annotations

import torch
from torch import nn

from ai.old_ai.policy_dataset import policy_output_size


INPUT_CHANNELS = 44


class PolicyModel(nn.Module):
    def __init__(self, output_size: int | None = None):
        super().__init__()

        self.output_size = output_size or policy_output_size()

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
            nn.Linear(512, self.output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)


def create_policy_model(device: str | torch.device | None = None) -> PolicyModel:
    model = PolicyModel()

    if device is not None:
        model = model.to(device)

    return model