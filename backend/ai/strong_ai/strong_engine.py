from __future__ import annotations

from ai.old_ai.evaluator import PIECE_VALUES, evaluate_board, score_to_win_rate
from ai.strong_ai.cshogi_engine import CshogiCandidate, get_cshogi_legal_candidates
from shogi_core.board import ShogiBoard, Move
from shogi_core.move_encoder import move_to_dict_with_id


ROOT_LIMIT = 26
REPLY_LIMIT = 32
THIRD_LIMIT = 16


class StrongEngine:
    def apply_move_copy(
        self,
        shogi: ShogiBoard,
        move: Move,
        turn: str,
    ) -> ShogiBoard:
        copied = shogi.clone()
        copied.turn = turn
        copied.apply_move(move)
        return copied

    def tactical_order_score(
        self,
        shogi: ShogiBoard,
        candidate: CshogiCandidate,
        turn: str,
    ) -> int:
        move = candidate.move
        score = 0

        if not move.drop:
            target = shogi.board[move.to_row][move.to_col]
            if target is not None and target["owner"] != turn:
                score += PIECE_VALUES.get(target["type"], 0) * 10

        if move.promote:
            score += 700

        if move.drop:
            score += 80

        try:
            after = self.apply_move_copy(shogi, move, turn)
            opponent = after.turn
            if after.is_in_check(opponent):
                score += 500
        except Exception:
            pass

        return score

    def ordered_candidates(
        self,
        shogi: ShogiBoard,
        turn: str,
        limit: int,
    ) -> list[CshogiCandidate]:
        candidates = get_cshogi_legal_candidates(shogi, turn)

        candidates.sort(
            key=lambda c: self.tactical_order_score(shogi, c, turn),
            reverse=True,
        )

        if limit > 0:
            return candidates[:limit]

        return candidates

    def search_after_reply(
        self,
        after_ai: ShogiBoard,
        ai_owner: str,
        reply: CshogiCandidate,
    ) -> int:
        reply_turn = after_ai.turn

        after_reply = self.apply_move_copy(
            shogi=after_ai,
            move=reply.move,
            turn=reply_turn,
        )

        ai_turn = after_reply.turn
        ai_followups = self.ordered_candidates(
            shogi=after_reply,
            turn=ai_turn,
            limit=THIRD_LIMIT,
        )

        if not ai_followups:
            return evaluate_board(after_reply, ai_owner=ai_owner)

        best_score = -999999

        for follow in ai_followups:
            after_follow = self.apply_move_copy(
                shogi=after_reply,
                move=follow.move,
                turn=ai_turn,
            )

            score = evaluate_board(after_follow, ai_owner=ai_owner)

            if score > best_score:
                best_score = score

        return int(best_score)

    def evaluate_candidate_3ply(
        self,
        shogi: ShogiBoard,
        candidate: CshogiCandidate,
        turn: str,
    ) -> tuple[int, int, str | None]:
        after_ai = self.apply_move_copy(
            shogi=shogi,
            move=candidate.move,
            turn=turn,
        )

        raw_score = evaluate_board(after_ai, ai_owner=turn)

        reply_turn = after_ai.turn
        replies = self.ordered_candidates(
            shogi=after_ai,
            turn=reply_turn,
            limit=REPLY_LIMIT,
        )

        if not replies:
            return raw_score + 999999, raw_score, None

        worst_score = 999999
        worst_reply_usi: str | None = None

        for reply in replies:
            score = self.search_after_reply(
                after_ai=after_ai,
                ai_owner=turn,
                reply=reply,
            )

            if score < worst_score:
                worst_score = score
                worst_reply_usi = reply.usi

        return int(worst_score), int(raw_score), worst_reply_usi

    def evaluate_candidates(
        self,
        shogi: ShogiBoard,
        turn: str,
        limit: int = 30,
    ) -> list[CshogiCandidate]:
        candidates = self.ordered_candidates(
            shogi=shogi,
            turn=turn,
            limit=ROOT_LIMIT,
        )

        for candidate in candidates:
            score, raw_score, worst_reply_usi = self.evaluate_candidate_3ply(
                shogi=shogi,
                candidate=candidate,
                turn=turn,
            )

            candidate.score = score
            candidate.raw_score = raw_score
            candidate.worst_reply_usi = worst_reply_usi
            candidate.order_score = self.tactical_order_score(shogi, candidate, turn)

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
        score = int(candidate.score)
        raw_score = int(getattr(candidate, "raw_score", score))

        return {
            "moveId": candidate.move_id,
            "usi": candidate.usi,
            "moveText": move_to_dict_with_id(candidate.move).get("moveText", candidate.move_id),
            "score": score,
            "searchScore": score,
            "rawScore": raw_score,
            "orderScore": int(getattr(candidate, "order_score", 0)),
            "winRate": score_to_win_rate(score),
            "worstReplyUsi": getattr(candidate, "worst_reply_usi", None),
            "move": candidate.move.to_dict(),
        }