# backend/ai/strong_ai/strong_engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StrongMove:
    move: Any
    score: float
    policy: float = 0.0
    value: float = 0.0
    is_best: bool = False


class StrongEngine:
    """
    cshogi / ShogiAIBook2 ベースの強いAIをここに接続する。
    まだ既存AIとは混ぜない。
    """

    def get_candidates(self, position: dict, limit: int = 20) -> list[StrongMove]:
        raise NotImplementedError("次の段階で cshogi ベースの強いAIを実装する")

    def select_best_move(self, position: dict) -> StrongMove:
        candidates = self.get_candidates(position, limit=20)

        if not candidates:
            raise ValueError("候補手がありません")

        candidates.sort(key=lambda x: x.score, reverse=True)
        candidates[0].is_best = True
        return candidates[0]
