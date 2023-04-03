#!/usr/env/python3
import torch
import os
from ai_old.util.chess import compact_board_enc_to_neural_board_enc, \
    compact_move_enc_to_neural_move_enc
from ai_old.nn.models.chess.move_pred import ResnetBasedMovePredictor


CTX_LEN = 12 # in ply
META_LEN = 7
BATCH_SIZE = 32
OPT_TYPE = 'adam'
LR = 0.001
N_ITER = 25000000


class MovePredDataset(torch.utils.data.Dataset):
    def __init__(self):
        dir = os.path.join(c.ASI_DATASETS_PATH, 'chess', 'maia', '2019-01')
        self.board = np.load(os.path.join(dir, 'board.npy'))
        self.move = np.load(os.path.join(dir, 'move.npy'))
        self.meta = np.load(os.path.join(dir, 'meta.npy'))
        print(f'[DATASET] board: {board.shape}')
        print(f'[DATASET] move: {move.shape}')
        print(f'[DATASET] meta: {meta.shape}')

        self.valid_idxs = []
        for i, x in enumerate(self.meta):
            move_ply = x[0]
            if move_ply >= CTX_LEN - 1:
                self.valid_idxs.append(i)

    def __len__(self):
        return len(self.valid_idxs)

    def __getitem__(self, idx):
        i = self.valid_idxs[idx]
        white_elo = self.meta[i, 1] # for sanity check
        boards = []
        for j in range(CTX_LEN):
            assert white_elo == self.meta[i-j, 1] # sanity
            boards.append(torch.tensor(self.board[i-j]).unsqueeze(0))
        return {
            'board': torch.cat(boards, dim=0),
            'meta': torch.tensor(self.meta[i]),
            'move': torch.tensor(self.move[i]),
        }


def get_dataset():
    return torch.utils.data.DataLoader(
        MoveDataset(),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )


def prep_batch(data):
    ctx = compact_board_enc_to_neural_board_enc(
        data['board'],
        data['meta'],
    ).to('cuda').to(torch.float32)
    move = compact_move_enc_to_neural_move_enc(data['move']).to('cuda')
    return ctx, move



def run():
    ds = get_dataset()

    model = ResnetBasedMovePredictor(None).train().to('cuda')

    if OPT_TYPE == 'adam':
        opt = torch.optim.Adam(enc_lerp_lc.parameters(), lr=LR)
    elif OPT_TYPE == 'ranger':
        opt = Ranger(enc_lerp_lc.parameters(), lr=LR)
    else:
        raise Exception(OPT_TYPE)

    loss_fn = nn.CrossEntropyLoss()

    for i in range(N_ITER):
        opt.zero_grad()
        batch = next(ds)
        ctx, move = prep_batch(batch)
        pred = model(ctx.detach())
        loss.backward()
        opt.step()


if __name__ == '__main__':
    run()
