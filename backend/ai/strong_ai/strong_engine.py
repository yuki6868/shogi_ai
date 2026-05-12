from __future__ import annotations

from ai.old_ai.evaluator import evaluate_board, score_to_win_rate
from ai.strong_ai.cshogi_engine import CshogiCandidate, get_cshogi_legal_candidates
from shogi_core.board import ShogiBoard
from shogi_core.move_encoder import move_to_dict_with_id


class StrongEngine:
    def evaluate_one_ply(
        self,
        shogi: ShogiBoard,
        candidate: CshogiCandidate,
        turn: str,
    ) -> int:
        copied = shogi.clone()
        copied.turn = turn
        copied.apply_move(candidate.move)

        return evaluate_board(copied, ai_owner=turn)

    def evaluate_with_reply(
        self,
        shogi: ShogiBoard,
        candidate: CshogiCandidate,
        turn: str,
        reply_limit: int = 40,
    ) -> tuple[int, int, str | None]:
        after_ai = shogi.clone()
        after_ai.turn = turn
        after_ai.apply_move(candidate.move)

        raw_score = evaluate_board(after_ai, ai_owner=turn)

        reply_turn = after_ai.turn
        replies = get_cshogi_legal_candidates(after_ai, reply_turn)

        if not replies:
            return raw_score + 99999, raw_score, None

        worst_score = 999999
        worst_usi: str | None = None

        for reply in replies[:reply_limit]:
            after_reply = after_ai.clone()
            after_reply.turn = reply_turn
            after_reply.apply_move(reply.move)

            score = evaluate_board(after_reply, ai_owner=turn)

            if score < worst_score:
                worst_score = score
                worst_usi = reply.usi

        return int(worst_score), int(raw_score), worst_usi

    def evaluate_candidates(
        self,
        shogi: ShogiBoard,
        turn: str,
        limit: int = 30,
    ) -> list[CshogiCandidate]:
        candidates = get_cshogi_legal_candidates(shogi, turn)

        for candidate in candidates:
            score, raw_score, worst_reply_usi = self.evaluate_with_reply(
                shogi=shogi,
                candidate=candidate,
                turn=turn,
            )

            candidate.score = score
            candidate.raw_score = raw_score
            candidate.worst_reply_usi = worst_reply_usi

        candidates.sort(key=lambda item: item.score, reverse=True)

        if limit > 0:
            return candidates[:limit]

        return candidates

    def get_candidates(
        self,
        shogi: ShogiBoard,
        turn: str,
        limit: int = 30,
    ) -> list[CshogiCandidate]:
        return self.evaluate_candidates(shogi, turn, limit)

    def select_best_move(
        self,
        shogi: ShogiBoard,
        turn: str,
    ) -> CshogiCandidate:
        candidates = self.evaluate_candidates(shogi, turn, limit=0)

        if not candidates:
            raise ValueError("候補手がありません")

        candidates[0].is_best = True
        return candidates[0]

    def candidate_to_dict(self, candidate: CshogiCandidate) -> dict:
        raw_score = int(getattr(candidate, "raw_score", candidate.score))
        score = int(candidate.score)

        return {
            "moveId": candidate.move_id,
            "usi": candidate.usi,
            "moveText": move_to_dict_with_id(candidate.move).get("moveText", candidate.move_id),
            "score": score,
            "rawScore": raw_score,
            "winRate": score_to_win_rate(score),
            "worstReplyUsi": getattr(candidate, "worst_reply_usi", None),
            "move": candidate.move.to_dict(),
        }