# backend/ai/mcts.py

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from ai.board import Move, ShogiBoard
from ai.evaluator import evaluate_board, score_to_win_rate
from ai.move_encoder import move_to_id
from ai.policy_dummy import filter_policy_candidates
from ai.policy_inference import get_policy_inference
from ai.value_inference import get_value_inference


MCTS_SIMULATIONS = 50
MCTS_DEPTH_LIMIT = 20
MAX_CANDIDATES = 8
CPUCT = 1.5
VALUE_SCALE = 1200.0


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


def _softmax_priors(policy_items: list[dict], legal_moves: list[Move]) -> list[tuple[Move, float]]:
    if not policy_items:
        n = max(1, len(legal_moves))
        return [(move, 1.0 / n) for move in legal_moves]

    max_score = max(float(item["policyScore"]) for item in policy_items)
    exp_values = [
        math.exp(float(item["policyScore"]) - max_score)
        for item in policy_items
    ]
    total = sum(exp_values) or 1.0

    return [
        (item["move"], exp_value / total)
        for item, exp_value in zip(policy_items, exp_values)
    ]


def _rank_candidates(board: ShogiBoard, legal_moves: list[Move], limit: int) -> list[tuple[Move, float]]:
    if not legal_moves:
        return []

    policy = get_policy_inference()

    if policy.available:
        ranked = policy.rank_legal_moves(
            shogi=board,
            legal_moves=legal_moves,
            top_k=limit,
        )
        priors = _softmax_priors(ranked, legal_moves)
        if priors:
            return priors

    candidates = filter_policy_candidates(
        board,
        legal_moves,
        owner=board.turn,
        top_k=limit,
    )

    if not candidates:
        candidates = legal_moves[:limit]

    weights = [float(len(candidates) - i) for i in range(len(candidates))]
    total = sum(weights) or 1.0

    return [
        (move, weight / total)
        for move, weight in zip(candidates, weights)
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


def run_mcts(
    shogi: ShogiBoard,
    root_owner: str = "enemy",
    simulations: int = MCTS_SIMULATIONS,
    depth_limit: int = MCTS_DEPTH_LIMIT,
) -> list[dict]:
    root_board = shogi.clone()
    root_board.turn = root_owner

    root = MCTSNode(board=root_board)
    _expand(root)

    if not root.children:
        return []

    for _ in range(max(1, simulations)):
        node = root
        path = [node]
        depth = 0

        while node.expanded and node.children and depth < depth_limit:
            node = _select_child(node, root_owner=root_owner)
            path.append(node)
            depth += 1

        if depth < depth_limit:
            terminal = _terminal_value(node.board, root_owner)
            if terminal is None:
                _expand(node)

        value = _evaluate(node, root_owner=root_owner)
        _backpropagate(path, value)

    results: list[dict] = []

    for child in root.children.values():
        q = child.q_value

        search_score = _value_to_score(q)

        # AIが1手指した直後の評価値
        # 画面に出す評価値はこっち
        raw_score = evaluate_board(child.board, ai_owner=root_owner)

        results.append(
            {
                "move": child.move,

                # 教育制御・画面表示用
                "score": raw_score,
                "rawScore": raw_score,
                "winRate": score_to_win_rate(raw_score),

                # MCTS内部探索結果
                "searchScore": search_score,
                "searchWinRate": score_to_win_rate(search_score),

                "visitCount": child.visit_count,
                "qValue": round(q, 4),
                "prior": round(child.prior, 4),
            }
        )

    results.sort(
        key=lambda item: (item["visitCount"], item["score"]),
        reverse=True,
    )

    return results