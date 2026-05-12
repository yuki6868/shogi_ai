# backend/ai/mcts.py

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from shogi_core.board import Move, ShogiBoard
from ai.old_ai.evaluator import evaluate_board, score_to_win_rate
from shogi_core.move_encoder import move_to_id
from ai.old_ai.policy_inference import get_policy_inference
from ai.old_ai.value_inference import get_value_inference


MCTS_SIMULATIONS = 0
MCTS_DEPTH_LIMIT = 3

MAX_CANDIDATES = 24
TACTICAL_CANDIDATE_LIMIT = 44
MINIMAX_BRANCH_LIMIT = 6

ROOT_DEEP_SEARCH_LIMIT = 12
FORCING_MOVE_LIMIT = 8

CPUCT = 1.5
VALUE_SCALE = 1200.0


TACTICAL_VALUES = {
    "P": 100,
    "L": 300,
    "N": 300,
    "S": 500,
    "G": 600,
    "B": 800,
    "R": 1000,
    "K": 10000,
    "+P": 600,
    "+L": 600,
    "+N": 600,
    "+S": 600,
    "+B": 1100,
    "+R": 1300,
}


@dataclass
class MCTSNode:
    board: ShogiBoard
    parent: Optional["MCTSNode"] = None
    move: Optional[Move] = None
    prior: float = 1.0
    children: dict[str, "MCTSNode"] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    expanded: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def _score_to_value(score: int) -> float:
    return math.tanh(score / VALUE_SCALE)


def _value_to_score(value: float) -> int:
    return int(max(-1.0, min(1.0, value)) * VALUE_SCALE)


def _terminal_value(board: ShogiBoard, root_owner: str) -> Optional[float]:
    legal_moves = board.generate_legal_moves(board.turn)
    if legal_moves:
        return None

    if board.turn == root_owner:
        return -1.0

    return 1.0


def _find_king(board: ShogiBoard, owner: str) -> Optional[tuple[int, int]]:
    for r in range(9):
        for c in range(9):
            piece = board.board[r][c]
            if piece and piece["owner"] == owner and piece["type"] == "K":
                return r, c
    return None


def _count_material_on_board(board: ShogiBoard) -> int:
    count = 0
    for r in range(9):
        for c in range(9):
            if board.board[r][c] is not None:
                count += 1
    return count


def _is_opening(board: ShogiBoard) -> bool:
    # 駒がほとんど減っていない間は序盤扱い
    return _count_material_on_board(board) >= 36


def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _king_safety_after_move(board: ShogiBoard, owner: str) -> float:
    king_pos = _find_king(board, owner)
    if king_pos is None:
        return 0.0

    score = 0.0
    kr, kc = king_pos

    # 玉の周囲に金銀がいると加点
    for r in range(max(0, kr - 2), min(9, kr + 3)):
        for c in range(max(0, kc - 2), min(9, kc + 3)):
            piece = board.board[r][c]
            if not piece:
                continue

            if piece["owner"] != owner:
                continue

            if piece["type"] in ("G", "S"):
                d = _distance(king_pos, (r, c))
                if d <= 1:
                    score += 220.0
                elif d <= 2:
                    score += 120.0

    # 玉が中央に残りすぎるのを少し嫌う
    # 先手側・後手側どちらでも左右に寄るほど加点
    if kc <= 2 or kc >= 6:
        score += 180.0
    elif kc == 3 or kc == 5:
        score += 80.0
    else:
        score -= 120.0

    return score


def _opening_score(board: ShogiBoard, move: Move, owner: str) -> float:
    if not _is_opening(board):
        return 0.0

    score = 0.0

    piece = None
    if not move.drop and move.from_row is not None and move.from_col is not None:
        piece = board.board[move.from_row][move.from_col]

    piece_type = move.piece if move.drop else (piece["type"] if piece else None)

    # 序盤の駒打ちは基本少し減点
    if move.drop:
        score -= 350.0

    # 序盤の飛角の突撃を少し抑える
    if (
        not move.drop
        and piece_type in ("B", "R")
        and move.from_row is not None
        and move.from_col is not None
    ):
        moved_distance = abs(move.to_row - move.from_row) + abs(move.to_col - move.from_col)
        if moved_distance >= 3:
            score -= 250.0

    # 飛角の突撃しすぎを抑える
    if (
        not move.drop
        and piece_type in ("B", "R")
        and move.from_row is not None
        and move.from_col is not None
    ):
        moved_distance = (
            abs(move.to_row - move.from_row)
            + abs(move.to_col - move.from_col)
        )

        # 序盤で深く突っ込む
        if moved_distance >= 4:
            copied = board.clone()
            copied.turn = owner

            try:
                copied.apply_move(move)

                enemy = copied.enemy_of(owner)

                # 相手に取られる可能性
                enemy_moves = copied.generate_legal_moves(enemy)

                attacked = False

                for em in enemy_moves:
                    if (
                        em.to_row == move.to_row
                        and em.to_col == move.to_col
                    ):
                        attacked = True
                        break

                if attacked:
                    score -= 500.0

            except Exception:
                pass

    # 歩を自然に進めるのは少し加点
    if (
        not move.drop
        and piece_type == "P"
        and move.from_col is not None
    ):
        score += 120.0

        # 角道っぽい歩
        if move.from_col in (2, 6):
            score += 180.0

        # 飛車先っぽい歩
        if move.from_col in (1, 7):
            score += 160.0

    # 金銀を前に出す・玉の近くに寄せる
    if (
        not move.drop
        and piece_type in ("G", "S")
        and move.from_row is not None
        and move.from_col is not None
    ):
        score += 180.0

        king_pos = _find_king(board, owner)
        if king_pos is not None:
            before = _distance(king_pos, (move.from_row, move.from_col))
            after = _distance(king_pos, (move.to_row, move.to_col))

            if after < before:
                score += 260.0
            elif after > before:
                score -= 120.0

    # 玉を左右に寄せる手を少し加点
    if (
        not move.drop
        and piece_type == "K"
        and move.from_col is not None
    ):
        center_col = 4
        before_side = abs(move.from_col - center_col)
        after_side = abs(move.to_col - center_col)

        if after_side > before_side:
            score += 420.0
        elif after_side < before_side:
            score -= 180.0

    # 指した後の玉の安全性
    copied = board.clone()
    copied.turn = owner
    try:
        copied.apply_move(move)
        score += _king_safety_after_move(copied, owner)
    except Exception:
        score -= 999999.0

    return score


def _tactical_score(
    board: ShogiBoard,
    move: Move,
    owner: str,
) -> float:
    score = 0.0

    captured = board.board[move.to_row][move.to_col]
    # タダ取り超優先
    if captured is not None and captured["owner"] != owner:
        piece_value = TACTICAL_VALUES.get(
            captured["type"],
            0,
        )

        score += piece_value * 2.5

    if captured is not None and captured["owner"] != owner:
        score += TACTICAL_VALUES.get(captured["type"], 0) * 5.0

    # 成りは強いが、前みたいに過剰評価しない
    if move.promote:
        score += 400.0

    # 序盤補正
    score += _opening_score(board, move, owner)

    copied = board.clone()
    copied.turn = owner

    try:
        before_score = evaluate_board(
            board,
            ai_owner=owner,
        )
        copied.apply_move(move)

        one_ply_score = evaluate_board(
            copied,
            ai_owner=owner,
        )

        # 1手後評価は補助程度
        score += one_ply_score * 0.05

        # 王手評価
        # 無条件王手加点は禁止。
        # 詰み・駒得・成り・評価改善がある王手だけ加点する。
        enemy = copied.enemy_of(owner)

        if copied.is_in_check(enemy):
            enemy_replies = copied.generate_legal_moves(enemy)

            if not enemy_replies:
                # 詰み王手
                score += 999999.0
            else:
                is_capture = (
                    captured is not None
                    and captured["owner"] != owner
                )

                is_good_check = (
                    is_capture
                    or move.promote
                    or one_ply_score >= before_score + 300
                )

                if is_good_check:
                    score += 180.0
                else:
                    # 無駄王手は明確に減点
                    score -= 1500.0

        # 成り超優先
        if move.promote:
            score += 120.0

    except Exception:
        score -= 999999.0

    return score


def _is_forcing_move(board: ShogiBoard, move: Move, owner: str) -> bool:
    captured = board.board[move.to_row][move.to_col]

    if captured is not None and captured["owner"] != owner:
        return True

    if move.promote:
        return True

    copied = board.clone()
    copied.turn = owner

    try:
        copied.apply_move(move)
        enemy = copied.enemy_of(owner)
        if copied.is_in_check(enemy):
            return False
    except Exception:
        return False

    return False

def _rank_candidates(
    board: ShogiBoard,
    legal_moves: list[Move],
    limit: int,
) -> list[tuple[Move, float]]:
    if not legal_moves:
        return []

    owner = board.turn
    scored: list[dict] = []

    for move in legal_moves:
        move_id = move_to_id(move)

        scored.append(
            {
                "move": move,
                "moveId": move_id,
                "score": _tactical_score(
                    board,
                    move,
                    owner,
                ),
            }
        )

    policy = get_policy_inference()

    if policy.available:
        try:
            ranked = policy.rank_legal_moves(
                shogi=board,
                legal_moves=legal_moves,
                top_k=len(legal_moves),
            )

            policy_score_map = {
                item["moveId"]: float(item["policyScore"])
                for item in ranked
            }

            for item in scored:
                policy_bonus = policy_score_map.get(
                    item["moveId"],
                    0.0,
                )

                # policyは補助程度にする
                item["score"] += policy_bonus * 120.0

        except Exception as e:
            print("[MCTS] policy ranking failed:", e)

    scored.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    forcing = [
        item for item in scored
        if _is_forcing_move(board, item["move"], owner)
    ]

    normal = [
        item for item in scored
        if item not in forcing
    ]

    selected = []

    selected.extend(forcing[:FORCING_MOVE_LIMIT])

    for item in normal:
        if len(selected) >= limit:
            break
        selected.append(item)

    total = sum(
        max(abs(item["score"]), 1.0)
        for item in selected
    ) or 1.0

    return [
        (
            item["move"],
            max(item["score"], 1.0) / total,
        )
        for item in selected
    ]


def _expand(node: MCTSNode) -> None:
    legal_moves = node.board.generate_legal_moves(node.board.turn)

    candidates = _rank_candidates(
        board=node.board,
        legal_moves=legal_moves,
        limit=MAX_CANDIDATES,
    )

    for move, prior in candidates:
        move_id = move_to_id(move)
        if move_id in node.children:
            continue

        next_board = node.board.clone()
        next_board.turn = node.board.turn
        next_board.apply_move(move)

        node.children[move_id] = MCTSNode(
            board=next_board,
            parent=node,
            move=move,
            prior=float(prior),
        )

    node.expanded = True


def _evaluate(node: MCTSNode, root_owner: str) -> float:
    terminal = _terminal_value(node.board, root_owner)
    if terminal is not None:
        return terminal

    value_ai = get_value_inference()

    if value_ai.available:
        try:
            return value_ai.predict_value_for_owner(
                node.board,
                owner=root_owner,
            )
        except Exception as e:
            print("[MCTS] value inference failed:", e)

    score = evaluate_board(node.board, ai_owner=root_owner)
    return _score_to_value(score)


def _select_child(node: MCTSNode, root_owner: str) -> MCTSNode:
    sign = 1.0 if node.board.turn == root_owner else -1.0
    parent_visits = max(1, node.visit_count)

    def puct(child: MCTSNode) -> float:
        q = child.q_value
        u = CPUCT * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
        noise = random.uniform(-1e-6, 1e-6)
        return sign * q + u + noise

    return max(node.children.values(), key=puct)


def _backpropagate(path: list[MCTSNode], value: float) -> None:
    for node in path:
        node.visit_count += 1
        node.value_sum += value


def _minimax_score(
    board: ShogiBoard,
    depth: int,
    alpha: int,
    beta: int,
    root_owner: str,
) -> int:
    legal_moves = board.generate_legal_moves(board.turn)

    if not legal_moves:
        if board.turn == root_owner:
            return -999999
        return 999999

    if depth <= 0:
        return evaluate_board(board, ai_owner=root_owner)

    candidates = _rank_candidates(
        board=board,
        legal_moves=legal_moves,
        limit=MINIMAX_BRANCH_LIMIT,
    )

    if board.turn == root_owner:
        best = -999999

        for move, _prior in candidates:
            copied = board.clone()
            copied.turn = board.turn
            copied.apply_move(move)

            score = _minimax_score(
                copied,
                depth - 1,
                alpha,
                beta,
                root_owner,
            )

            best = max(best, score)
            alpha = max(alpha, best)

            if beta <= alpha:
                break

        return best

    best = 999999

    for move, _prior in candidates:
        copied = board.clone()
        copied.turn = board.turn
        copied.apply_move(move)

        score = _minimax_score(
            copied,
            depth - 1,
            alpha,
            beta,
            root_owner,
        )

        best = min(best, score)
        beta = min(beta, best)

        if beta <= alpha:
            break

    return best

def run_mcts(
    shogi: ShogiBoard,
    root_owner: str = "enemy",
    simulations: int = MCTS_SIMULATIONS,
    depth_limit: int = MCTS_DEPTH_LIMIT,
) -> list[dict]:
    root_board = shogi.clone()
    root_board.turn = root_owner

    legal_moves = root_board.generate_legal_moves(root_owner)

    if not legal_moves:
        return []
    
    current_root_score = evaluate_board(root_board, ai_owner=root_owner)

    root_candidates = _rank_candidates(
        board=root_board,
        legal_moves=legal_moves,
        limit=MAX_CANDIDATES,
    )

    results: list[dict] = []

    for index, (move, prior) in enumerate(root_candidates):
        copied = root_board.clone()
        copied.turn = root_owner
        copied.apply_move(move)

        raw_score = evaluate_board(copied, ai_owner=root_owner)

        if index < ROOT_DEEP_SEARCH_LIMIT:
            search_score = _minimax_score(
                board=copied,
                depth=max(1, min(3, depth_limit) - 1),
                alpha=-999999,
                beta=999999,
                root_owner=root_owner,
            )
        else:
            search_score = raw_score

        enemy = copied.enemy_of(root_owner)
        is_check = copied.is_in_check(enemy)

        enemy_replies = copied.generate_legal_moves(enemy) if is_check else []
        is_mate_check = is_check and not enemy_replies

        captured = root_board.board[move.to_row][move.to_col]
        is_capture = captured is not None and captured["owner"] != root_owner

        # 良い王手判定は rawScore ではなく searchScore で見る
        # rawScore は「王手した瞬間」だけ高くなりやすいので信用しない
        is_good_check = (
            is_mate_check
            or is_capture
            or move.promote
            or search_score >= current_root_score + 300
        )

        # 無駄王手は searchScore 自体を落とす
        # ここで落とさないと、教育選択側でまだ選ばれる
        if is_check and not is_good_check:
            search_score -= 1200
            raw_score -= 800

        results.append(
            {
                "move": move,
                "score": int(search_score),
                "searchScore": int(search_score),
                "winRate": score_to_win_rate(int(search_score)),
                "searchWinRate": score_to_win_rate(int(search_score)),

                "rawScore": int(raw_score),
                "rawWinRate": score_to_win_rate(int(raw_score)),

                "visitCount": max(1, len(root_candidates) - index),
                "qValue": round(_score_to_value(int(search_score)), 4),
                "prior": round(float(prior), 4),
                "isCheck": bool(is_check),
                "isGoodCheck": bool(is_good_check),
                "isMateCheck": bool(is_mate_check),
            }
        )


    results.sort(
        key=lambda item: (
            int(item["searchScore"]),
            int(item["rawScore"]),
            int(item["visitCount"]),
        ),
        reverse=True,
    )

    return results