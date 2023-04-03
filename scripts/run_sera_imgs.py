#!/usr/bin/env python3
from PIL import Image
import numpy as np
from ai_old.util.face import get_faces, get_landmarks, align_face
import os
from ai_old.util.factory import build_model_from_exp
from ai_old.util.pretrained import build_pretrained_e4e, build_pretrained_sg2
import torch
from ai_old.util.etc import normalized_tensor_to_pil_img, create_img_row, \
    resize_imgs, create_img_grid
from ai_old.nn.models.swap.w_plus import SwapperAndPtiGenerator


ID_TO_IDX = [
    0, 1, 0, 2, # 0-3
    None, 0, 0, 1, # 4-7
    0, 1, 0, None, # 8-11
    0, 1, 2, 2, # 12-15
]

FOLDER = '/home/asiu/data/sera'
MAGS = [0., 0.5, 0.75, 1., 1.25, 1.5]

ID_TO_BEST_MAG = [
    .5, .75, 1.25, 1., # 0-3
    None, .5, .5, .5, # 4-7
    .5, .75, 1., None, # 8-11
    .75, .75, 1., .75, # 12-15
]

PTI_ID_TO_BEST_MAG = [
    .75, .5, .75, .75, # 0-3
    None, .75, .75, .5, # 4-7
    .5, .75, 1., None, # 8-11
    .75, .5, 1., .5, # 12-15
]


def align():
    for filename in os.listdir(os.path.join(FOLDER, 'og')):
        id = int(filename.split('.')[0])
        idx = ID_TO_IDX[id]
        if idx is None:
            continue

        print(id)
        img = Image.open(os.path.join(FOLDER, 'og', filename))
        img_np = np.asarray(img)
        box = get_faces(img_np)[idx]
        landmarks = get_landmarks(img_np, box)

        aligned = align_face(img, landmarks, 256)[0]
        aligned.save(os.path.join(FOLDER, 'aligned', '256', f'{id}.png'))

        aligned = align_face(img, landmarks, 1024)[0]
        aligned.save(os.path.join(FOLDER, 'aligned', '1024', f'{id}.png'))


def invert():
    E = build_pretrained_e4e()
    G = build_pretrained_sg2().synthesis
    for filename in os.listdir(os.path.join(FOLDER, 'aligned', '256')):
        id = int(filename.split('.')[0])
        print(id)
        img = np.asarray(Image.open(os.path.join(
            FOLDER, 'aligned', '256', filename)))
        assert img.dtype == np.uint8
        img = torch.from_numpy(img.transpose(2, 0, 1))
        img = img.to('cuda').to(torch.float32) / 127.5 - 1

        ws = E(img.unsqueeze(0))
        invs = G(ws)

        inv = normalized_tensor_to_pil_img(invs[0])
        inv.save(os.path.join(FOLDER, 'inverted', f'{id}.png'))

        np.save(
            os.path.join(FOLDER, 'w', f'{id}.npy'),
            ws[0].cpu().numpy(),
        )


def swap():
    G = build_model_from_exp('wswap/1/0', 'G', return_cfg=False).to('cuda')
    G.eval()

    id_to_og_and_best = {}

    for filename in os.listdir(os.path.join(FOLDER, 'w')):
        id = int(filename.split('.')[0])
        print(id)
        gender = torch.ones(1, 1, dtype=torch.float32, device='cuda')
        w = torch.from_numpy(
            np.load(os.path.join(FOLDER, 'w', f'{id}.npy')),
        ).unsqueeze(0).to('cuda').to(torch.float32)

        swaps = []
        best = None
        for mag in MAGS:
            swap_w, delta = G.f(w, gender, magnitude=mag)
            swap_img = G.g(swap_w)
            swaps.append(resize_imgs(swap_img, 256)[0])
            if mag == ID_TO_BEST_MAG[id]:
                best = len(swaps) - 1

        swaps = [normalized_tensor_to_pil_img(x) for x in swaps]
        og = Image.open(os.path.join(FOLDER, 'aligned', '256', f'{id}.png'))
        id_to_og_and_best[id] = (og, swaps[best])

        create_img_row([og] + swaps, 256).save(
            os.path.join(FOLDER, 'progression', f'{id}.png'))
        create_img_row([og, swaps[best]], 256).save(
            os.path.join(FOLDER, 'best', f'{id}.png'))
        swaps[best].save(
            os.path.join(FOLDER, 'best-standalone', f'{id}.png'))

    best_grid = (
        id_to_og_and_best[14],
        id_to_og_and_best[12],
        id_to_og_and_best[15],
    )
    create_img_grid(best_grid, 256).save(os.path.join(FOLDER, 'best.png'))


def new_swap():
    G = build_model_from_exp('lerp/5/5', 'G', return_cfg=False).to('cuda')
    G.eval()

    id_to_og_and_best = {}

    for filename in os.listdir(os.path.join(FOLDER, 'w')):
        id = int(filename.split('.')[0])
        print(id)
        gender = torch.ones(1, 1, dtype=torch.float32, device='cuda')
        w = torch.from_numpy(
            np.load(os.path.join(FOLDER, 'w', f'{id}.npy')),
        ).unsqueeze(0).to('cuda').to(torch.float32)

        swaps = []
        best = None
        for mag in MAGS:
            swap_w = G.f(w, gender, magnitude=mag)
            swap_img = G.g(swap_w)
            swaps.append(resize_imgs(swap_img, 256)[0])
            if mag == ID_TO_BEST_MAG[id]:
                best = len(swaps) - 1

        swaps = [normalized_tensor_to_pil_img(x) for x in swaps]
        og = Image.open(os.path.join(FOLDER, 'aligned', '256', f'{id}.png'))
        id_to_og_and_best[id] = (og, swaps[best])

        create_img_row([og] + swaps, 256).save(
            os.path.join(FOLDER, 'new-progression', f'{id}.png'))
        create_img_row([og, swaps[best]], 256).save(
            os.path.join(FOLDER, 'new-best', f'{id}.png'))
        swaps[best].save(
            os.path.join(FOLDER, 'new-best-standalone', f'{id}.png'))

    best_grid = (
        id_to_og_and_best[14],
        id_to_og_and_best[12],
        id_to_og_and_best[15],
    )
    create_img_grid(best_grid, 256).save(os.path.join(FOLDER, 'new-best.png'))


def pti():
    for filename in os.listdir(os.path.join(FOLDER, 'aligned', '256')):
        id = int(filename.split('.')[0])
        print(id)
        gender = torch.ones(1, 1, dtype=torch.float32, device='cuda')
        w = torch.load(
            os.path.join(FOLDER, 'pti-w', str(id), '0.pt'),
        ).to('cuda').to(torch.float32)
        print(w.shape)

        G = SwapperAndPtiGenerator(os.path.join(
            FOLDER,
            'pti-models',
            f'{id}.pt',
        )).to('cuda')
        G.eval()

        swaps = []
        best = None
        for mag in MAGS:
            swap_w, delta = G.f(w, gender, magnitude=mag)
            swap_img = G.g(swap_w, noise_mode='const')
            swaps.append(resize_imgs(swap_img, 256)[0])
            if mag == ID_TO_BEST_MAG[id]:
                best = len(swaps) - 1

        swaps = [normalized_tensor_to_pil_img(x) for x in swaps]
        og = Image.open(os.path.join(FOLDER, 'aligned', '256', f'{id}.png'))

        create_img_row([og] + swaps, 256).save(
            os.path.join(FOLDER, 'pti-progression', f'{id}.png'))
        create_img_row([og, swaps[best]], 256).save(
            os.path.join(FOLDER, 'pti-best', f'{id}.png'))

        if id == 13:
            old = Image.open(
                os.path.join(FOLDER, 'best-standalone', f'{id}.png'))
            create_img_row([og, old, swaps[best]], 256).save(os.path.join(
                FOLDER, 'pti-comparison.png'))


def new_pti():
    for filename in os.listdir(os.path.join(FOLDER, 'aligned', '256')):
        id = int(filename.split('.')[0])
        print(id)
        gender = torch.ones(1, 1, dtype=torch.float32, device='cuda')
        w = torch.load(
            os.path.join(FOLDER, 'pti-w', str(id), '0.pt'),
        ).to('cuda').to(torch.float32)

        G = SwapperAndPtiGenerator(
            pti_g_path=os.path.join(
                FOLDER,
                'pti-models',
                f'{id}.pt',
            ),
            swap_exp='lerp/5/5',
        ).to('cuda')
        G.eval()

        swaps = []
        best = None
        for mag in MAGS:
            swap_w = G.f(w, gender, magnitude=mag)
            swap_img = G.g(swap_w, noise_mode='const')
            swaps.append(resize_imgs(swap_img, 256)[0])
            if mag == ID_TO_BEST_MAG[id]:
                best = len(swaps) - 1

        swaps = [normalized_tensor_to_pil_img(x) for x in swaps]
        og = Image.open(os.path.join(FOLDER, 'aligned', '256', f'{id}.png'))

        create_img_row([og] + swaps, 256).save(
            os.path.join(FOLDER, 'new-pti-progression', f'{id}.png'))
        create_img_row([og, swaps[best]], 256).save(
            os.path.join(FOLDER, 'new-pti-best', f'{id}.png'))

        if id == 13:
            old = Image.open(
                os.path.join(FOLDER, 'new-best-standalone', f'{id}.png'))
            create_img_row([og, old, swaps[best]], 256).save(os.path.join(
                FOLDER, 'new-pti-comparison.png'))


if __name__ == '__main__':
    with torch.no_grad():
        # align()
        # invert()
        # swap()
        # pti()
        # new_swap()
        new_pti()




























# tmp
