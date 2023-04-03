#!/usr/bin/env python3
import chess
import numpy as np
import ai_old.constants as c
import os


COL = [
    'game_id', 'type', 'result', 'white_player', 'black_player', 'white_elo',
    'black_elo', 'time_control', 'num_ply', 'termination', 'white_won',
    'black_won', 'no_winner', 'move_ply', 'move', 'cp', 'cp_rel', 'cp_loss',
    'is_blunder_cp', 'winrate', 'winrate_elo', 'winrate_loss', 'is_blunder_wr',
    'opp_winrate', 'white_active', 'active_elo', 'opponent_elo', 'active_won',
    'is_capture', 'clock', 'opp_clock', 'clock_percent', 'opp_clock_percent',
    'low_time', 'board', 'active_bishop_count', 'active_knight_count',
    'active_pawn_count', 'active_queen_count', 'active_rook_count', 'is_check',
    'num_legal_moves', 'opp_bishop_count', 'opp_knight_count', 'opp_pawn_count',
    'opp_queen_count', 'opp_rook_count',
]
COL = {x: i for i, x in enumerate(COL)}

ALL_TYPES = [
    'Blitz', 'Rapid', 'Bullet', 'Classical', 'UltraBullet', 'Correspondence']
INVALID_TYPES = ['Bullet', 'UltraBullet', 'Correspondence']

MAX_ELO = 3500

PIECE_ENC = [None, 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
PIECE_ENC = {x: i for i, x in enumerate(PIECE_ENC)}

PROMOTION_PIECE_ENC = [chess.QUEEN, chess.KNIGHT, chess.BISHOP, chess.ROOK]
PROMOTION_PIECE_ENC = {x: i for i, x in enumerate(PROMOTION_PIECE_ENC)}


def chess_square_to_int(square):
    return chess.square_rank(square) * 8 + chess.square_file(square)


def run(base_path, year_month):
    folder = os.path.join(base_path, year_month)
    board_data = [] # n x 8 x 8
    move_data = [] # n x 3
    metadata = [] # n x 7
    eval_data = [] # n floats
    i = 0
    with open(os.path.join(folder, 'raw.csv'), 'r') as f:
        next(f)
        prev_move_ply = None
        for line in f:
            print(i)

            # parse line
            line = line.strip().split(',')
            game_type = line[COL['type']]
            white_elo = int(line[COL['white_elo']])
            assert white_elo < MAX_ELO
            black_elo = int(line[COL['black_elo']])
            assert black_elo < MAX_ELO
            move_ply = int(line[COL['move_ply']])
            move_uci_str = line[COL['move']]
            # eval = float(line[COL['cp']])
            low_time = line[COL['low_time']]
            board_fen = line[COL['board']]

            # sanity
            if game_type != 'Correspondence':
                assert low_time in ['True', 'False']
            if game_type not in ALL_TYPES:
                raise Exception(line[COL['type']])
            if move_ply > 0:
                assert move_ply == prev_move_ply + 1
            assert white_elo < MAX_ELO
            assert black_elo < MAX_ELO

            # update state
            i += 1
            prev_move_ply = move_ply

            # skip low time situations
            if low_time == 'True' or game_type in INVALID_TYPES:
                continue

            # encode board
            board_obj = chess.Board(board_fen)
            board_enc = np.zeros((8, 8), dtype=np.uint8)
            for y in range(8):
                for x in range(8):
                    square = chess.square(x, y)
                    piece = board_obj.piece_at(square)
                    if piece is not None:
                        board_enc[y][x] = PIECE_ENC[piece.symbol()]
            board_data.append(board_enc)

            # encode move
            move_obj = chess.Move.from_uci(move_uci_str)
            move_enc = np.zeros((3,), dtype=np.uint8)
            move_enc[0] = chess_square_to_int(move_obj.from_square)
            move_enc[1] = chess_square_to_int(move_obj.to_square)
            if move_obj.promotion is not None:
                move_enc[2] = PROMOTION_PIECE_ENC[move_obj.promotion]
            move_data.append(move_enc)

            # encode metadata
            metadata_enc = np.zeros((7,), dtype=np.uint8)
            metadata_enc[0] = move_ply
            metadata_enc[1] = int(255. * white_elo / MAX_ELO)
            metadata_enc[2] = int(255. * black_elo / MAX_ELO)
            metadata_enc[3] = int(board_obj.has_kingside_castling_rights(
                chess.WHITE))
            metadata_enc[4] = int(board_obj.has_kingside_castling_rights(
                chess.BLACK))
            metadata_enc[5] = int(board_obj.has_queenside_castling_rights(
                chess.WHITE))
            metadata_enc[6] = int(board_obj.has_queenside_castling_rights(
                chess.BLACK))
            metadata.append(metadata_enc)

            # save engine eval
            # eval_data.append(eval)


    print('data len: ', len(board_data))
    assert len(board_data) == len(move_data) == len(metadata) == len(eval_data)

    np.save(
        os.path.join(folder, 'board.npy'),
        np.stack(board_data),
    )
    np.save(
        os.path.join(folder, 'move.npy'),
        np.stack(move_data),
    )
    np.save(
        os.path.join(folder, 'meta.npy'),
        np.stack(metadata),
    )
    # np.save(
    #     os.path.join(folder, 'eval.npy'),
    #     np.stack(eval_data),
    # )


if __name__ == '__main__':
    folder = os.path.join(c.ASI_DATASETS_PATH, 'chess', 'maia')
    run(folder, '2019-01')
