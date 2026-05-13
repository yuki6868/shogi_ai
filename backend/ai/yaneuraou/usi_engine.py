from __future__ import annotations

import os
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from ai.strong_ai.cshogi_engine import shogi_to_sfen, usi_to_move
from shogi_core.board import Move, ShogiBoard
from shogi_core.move_encoder import move_to_dict_with_id, move_to_id
from ai.old_ai.evaluator import score_to_win_rate

BASE_DIR = Path(__file__).resolve().parents[3]

DEFAULT_ENGINE_PATH = (
    BASE_DIR
    / "YaneuraOu"
    / "source"
    / "YaneuraOu-by-gcc"
)

INFO_RE = re.compile(
    r"\binfo\b.*?\bdepth\s+(?P<depth>\d+).*?"
    r"(?:\bmultipv\s+(?P<multipv>\d+)\s+)?"
    r"\bscore\s+(?P<score_type>cp|mate)\s+(?P<score>-?\d+).*?"
    r"\bpv\s+(?P<pv>\S+)(?P<rest>.*)$"
)


@dataclass
class YaneuraOuCandidate:
    usi: str
    move: Move
    move_id: str
    score: int
    depth: int
    multipv: int
    pv: list[str]
    is_best: bool = False


def _mate_to_score(value: int) -> int:
    if value > 0:
        return 100000 - value
    return -100000 - value


class YaneuraOuUSIEngine:
    def __init__(self, engine_path=None, threads: int = 2, multipv: int = 8):
        env_path = os.getenv("YANEURAOU_ENGINE_PATH")
        self.engine_path = Path(engine_path or env_path or DEFAULT_ENGINE_PATH).expanduser()
        self.threads = threads
        self.multipv = multipv
        self.proc = None
        self.lock = threading.Lock()

    def available(self) -> bool:
        return self.engine_path.exists() and os.access(self.engine_path, os.X_OK)

    def _start_locked(self):
        if self.proc is not None and self.proc.poll() is None:
            return

        if not self.available():
            raise FileNotFoundError(
                f"やねうら王が見つかりません: {self.engine_path}\n"
                "必要なら YANEURAOU_ENGINE_PATH を設定してください。"
            )

        self.proc = subprocess.Popen(
            [str(self.engine_path)],
            cwd=str(self.engine_path.parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        self._send_locked("usi")
        self._read_until_locked("usiok")

        self._send_locked(f"setoption name Threads value {self.threads}")
        self._send_locked(f"setoption name MultiPV value {self.multipv}")
        self._send_locked("isready")
        self._read_until_locked("readyok")

    def _send_locked(self, command: str):
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def _read_line_locked(self) -> str:
        line = self.proc.stdout.readline()
        if line == "":
            raise RuntimeError("USIエンジンが終了しました")
        return line.rstrip("\n")

    def _read_until_locked(self, marker: str):
        while True:
            line = self._read_line_locked()
            if line.strip() == marker or line.startswith(marker):
                return

    def _parse_info_line(self, shogi: ShogiBoard, line: str):
        match = INFO_RE.search(line)
        if not match:
            return None

        usi = match.group("pv")
        if usi in ("resign", "win", "none"):
            return None

        score_type = match.group("score_type")
        raw_score = int(match.group("score"))
        score = raw_score if score_type == "cp" else _mate_to_score(raw_score)

        depth = int(match.group("depth"))
        multipv = int(match.group("multipv") or 1)

        rest = match.group("rest").strip()
        pv = [usi] + (rest.split() if rest else [])

        try:
            move = usi_to_move(shogi, usi)
        except Exception:
            return None

        return YaneuraOuCandidate(
            usi=usi,
            move=move,
            move_id=move_to_id(move),
            score=score,
            depth=depth,
            multipv=multipv,
            pv=pv,
        )

    def analyze(self, shogi: ShogiBoard, turn: str, depth: int = 8, multipv: int = 8):
        with self.lock:
            self._start_locked()

            self._send_locked(f"setoption name MultiPV value {multipv}")
            self._send_locked("isready")
            self._read_until_locked("readyok")

            sfen = shogi_to_sfen(shogi, turn=turn)

            self._send_locked(f"position sfen {sfen}")
            self._send_locked(f"go depth {depth}")

            latest = {}
            bestmove_usi = None

            while True:
                line = self._read_line_locked()

                if line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) >= 2:
                        bestmove_usi = parts[1]
                    break

                candidate = self._parse_info_line(shogi, line)
                if candidate:
                    latest[candidate.multipv] = candidate

            candidates = [latest[k] for k in sorted(latest)]

            for c in candidates:
                c.is_best = c.usi == bestmove_usi

            if candidates and not any(c.is_best for c in candidates):
                candidates[0].is_best = True

            return candidates


def candidate_to_dict(candidate: YaneuraOuCandidate) -> dict:
    return {
        "moveId": candidate.move_id,
        "usi": candidate.usi,
        "moveText": move_to_dict_with_id(candidate.move).get("moveText", candidate.move_id),
        "score": int(candidate.score),
        "searchScore": int(candidate.score),
        "rawScore": int(candidate.score),
        "winRate": score_to_win_rate(int(candidate.score)),
        "depth": candidate.depth,
        "multipv": candidate.multipv,
        "pv": candidate.pv[:8],
        "isBest": candidate.is_best,
        "move": candidate.move.to_dict(),
    }


_yaneuraou_engine = None


def get_yaneuraou_engine() -> YaneuraOuUSIEngine:
    global _yaneuraou_engine
    if _yaneuraou_engine is None:
        _yaneuraou_engine = YaneuraOuUSIEngine()
    return _yaneuraou_engine