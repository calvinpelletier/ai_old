#!/usr/bin/env python3
import torch


DX_DY_TO_ONE_HOT_IDX = [
    [28  ,None,None,None,None,None,None,14  ,None,None,None,None,None,None,55  ],
    [None,29  ,None,None,None,None,None,15  ,None,None,None,None,None,54  ,None],
    [None,None,30  ,None,None,None,None,16  ,None,None,None,None,53  ,None,None],
    [None,None,None,31  ,None,None,None,17  ,None,None,None,52  ,None,None,None],
    [None,None,None,None,32  ,None,None,18  ,None,None,51  ,None,None,None,None],
    [None,None,None,None,None,33  ,63  ,19  ,56  ,50  ,None,None,None,None,None],
    [None,None,None,None,None,62  ,34  ,20  ,49  ,57  ,None,None,None,None,None],
    [0   ,1   ,2   ,3   ,4   ,5   ,6   ,None,7   ,8   ,9   ,10  ,11  ,12  ,13  ],
    [None,None,None,None,None,61  ,48  ,21  ,35  ,58  ,None,None,None,None,None],
    [None,None,None,None,None,47  ,60  ,22  ,59  ,36  ,None,None,None,None,None],
    [None,None,None,None,46  ,None,None,23  ,None,None,37  ,None,None,None,None],
    [None,None,None,45  ,None,None,None,24  ,None,None,None,38  ,None,None,None],
    [None,None,44  ,None,None,None,None,25  ,None,None,None,None,39  ,None,None],
    [None,43  ,None,None,None,None,None,26  ,None,None,None,None,None,40  ,None],
    [42  ,None,None,None,None,None,None,27  ,None,None,None,None,None,None,41  ],
]
STRAIGHT_UNDERPROMOTION = [None, 64, 65, 66]
RIGHT_UNDERPROMOTION = [None, 67, 68, 69]
LEFT_UNDERPROMOTION = [None, 70, 71, 72]


def int_to_chess_coords(chess_int):
    return chess_int % 8, chess_int // 8


def compact_move_enc_to_neural_move_enc(compact_move_enc):
    n = compact_move_enc.shape[0]
    out = torch.zeros(
        n, 73, 8, 8,
        device=compact_move_enc.device,
        dtype=torch.uint8,
    )
    for i in range(n):
        from_x, from_y = int_to_chess_coords(compact_move_enc[i][0])
        to_x, to_y = int_to_chess_coords(compact_move_enc[i][1])
        dx = to_x - from_x
        dy = to_y - from_y
        underpromotion = compact_move_enc[i][2]
        if underpromotion > 0:
            if dx == 0:
                idx = STRAIGHT_UNDERPROMOTION[underpromotion]
            elif dx > 0:
                idx = RIGHT_UNDERPROMOTION[underpromotion]
            else:
                idx = LEFT_UNDERPROMOTION[underpromotion]
        else:
            idx = DX_DY_TO_ONE_HOT_IDX[dx + 7][dy + 7]
        out[i][idx][from_y][from_x] = 1
    return out.reshape(n, 4672)


def compact_board_enc_to_neural_board_enc(compact_board_enc, compact_meta_enc):
    n, ctx_len, _, _ = compact_board_enc.shape
    assert n == compact_meta_enc.shape[0]
    device = compact_board_enc.device

    meta = torch.zeros(n, 7, device=device, dtype=torch.uint8)
    for i, x in enumerate(compact_meta_enc):
        meta[i][0] = x[0] % 2 # who's turn
        meta[i][1:] = x[1:7] # elos, castling rights

    board = torch.zeros(n, ctx_len * 12, 8, 8, device=device, dtype=torch.uint8)
    for i, ctx in enumerate(compact_board_enc):
        for y in range(8):
            for x in range(8):
                for j in range(ctx_len):
                    piece = ctx[j]
                    if piece > 0:
                        idx = j * 12 + (piece - 1)
                        board[i][idx][y][x] = 1

    meta = meta.reshape(n, 7, 1, 1).repeat(1, 1, 8, 8)
    out = torch.cat([meta, board], dim=1)
    print('out', out.shape)
    return out






























# tmp
