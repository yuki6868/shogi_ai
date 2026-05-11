from __future__ import annotations

from typing import Dict

import torch

from ai.board import BOARD_SIZE, ShogiBoard


PIECE_CHANNELS: Dict[str, int] = {
    "P": 0,
    "L": 1,
    "N": 2,
    "S": 3,
    "G": 4,
    "B": 5,
    "R": 6,
    "K": 7,
    "+P": 8,
    "+L": 9,
    "+N": 10,
    "+S": 11,
    "+B": 12,
    "+R": 13,
}


CHANNEL_COUNT = 28


def board_to_tensor(board: ShogiBoard) -> torch.Tensor:
    """
    shape:
        (28, 9, 9)

    0-13:
        自分の駒

    14-27:
        相手の駒
    """

    tensor = torch.zeros(
        (CHANNEL_COUNT, BOARD_SIZE, BOARD_SIZE),
        dtype=torch.float32,
    )

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board.board[row][col]

            if piece is None:
                continue

            piece_type = piece["type"]
            owner = piece["owner"]

            if piece_type not in PIECE_CHANNELS:
                continue

            channel = PIECE_CHANNELS[piece_type]

            if owner == "enemy":
                channel += 14

            tensor[channel, row, col] = 1.0

    return tensor


def board_to_tensor_with_turn(board: ShogiBoard) -> torch.Tensor:
    """
    手番込み版

    shape:
        (30, 9, 9)

    28:
        player turn

    29:
        enemy turn
    """

    base = board_to_tensor(board)

    turn_tensor = torch.zeros(
        (2, BOARD_SIZE, BOARD_SIZE),
        dtype=torch.float32,
    )

    if board.turn == "player":
        turn_tensor[0, :, :] = 1.0
    else:
        turn_tensor[1, :, :] = 1.0

    return torch.cat([base, turn_tensor], dim=0)


def hand_to_tensor(board: ShogiBoard) -> torch.Tensor:
    """
    持ち駒テンソル

    shape:
        (14,)
    """

    tensor = torch.zeros((14,), dtype=torch.float32)

    player_hand = board.player_hand
    enemy_hand = board.enemy_hand

    for piece in player_hand:
        if piece in PIECE_CHANNELS:
            tensor[PIECE_CHANNELS[piece]] += 1

    for piece in enemy_hand:
        if piece in PIECE_CHANNELS:
            tensor[PIECE_CHANNELS[piece] + 7] += 1

    return tensor


def board_to_full_tensor(board: ShogiBoard) -> torch.Tensor:
    """
    完全版テンソル

    内容:
        ・盤面
        ・手番
        ・持ち駒

    output:
        (44, 9, 9)

    0-27:
        盤面

    28-29:
        手番

    30-43:
        持ち駒
    """

    board_tensor = board_to_tensor_with_turn(board)

    hand_tensor_1d = hand_to_tensor(board)

    hand_tensor = torch.zeros(
        (14, BOARD_SIZE, BOARD_SIZE),
        dtype=torch.float32,
    )

    for i in range(14):
        hand_tensor[i, :, :] = hand_tensor_1d[i]

    return torch.cat([board_tensor, hand_tensor], dim=0)


if __name__ == "__main__":
    from ai.kifu_parser import create_initial_board

    board = create_initial_board()

    tensor = board_to_full_tensor(board)

    print("shape =", tensor.shape)
    print(tensor)