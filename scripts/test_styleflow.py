#!/usr/bin/env python3
import os
from PIL import Image
import torch
import numpy as np
from ai_old.util.factory import build_model_from_exp
from tqdm import tqdm
from external.styleflow.flow import cnf


NUM_WS = 14
IMSIZE = 256
# UNALTERED_WS_FROM_IDXS = [6, 8, 10, 12, 14]
# UNALTERED_WS_FROM_IDXS = [14]
G_EXP = 'facegen/9/1'
STYLEFLOW_EXP = 'e4e-facegen-9-1'
VERSIONS = ['0', '1', '2', '3', '4']
VERSION_TO_N_BLOCKS = {
    '0': 5,
    '1': 3,
    '2': 7,
    '3': 5,
    '4': 5,
}
VERSION_TO_N_ATTRS = {
    '0': 7,
    '1': 7,
    '2': 7,
    '3': 7,
    '4': 16,
}
OG_IMG_PATH = '/home/asiu/datasets/supplemental/face_image_256/ffhq-128/{}.png'


def run():
    G, _ = build_model_from_exp('facegen/9/1', 'G_ema')
    G = G.eval().requires_grad_(False).cuda()

    As = {}
    for version in VERSIONS:
        flow_modules = '-'.join(
            ['512' for _ in range(VERSION_TO_N_BLOCKS[version])]
        )
        A = cnf(512, flow_modules, VERSION_TO_N_ATTRS[version], 1)
        A.load_state_dict(torch.load(
            f'/home/asiu/data/styleflow/{STYLEFLOW_EXP}/{version}/model.pt'))
        As[version] = A.eval()

    latents_dir = f'/home/asiu/datasets/{STYLEFLOW_EXP}/latents'
    attributes_dir = f'/home/asiu/datasets/{STYLEFLOW_EXP}/attributes'
    save_dir = f'/home/asiu/datasets/{STYLEFLOW_EXP}/debug/swap'

    zero_pad = torch.zeros(1, NUM_WS, 1).cuda()

    for fname in tqdm(os.listdir(attributes_dir)):
        id = fname.split('.')[0]
        ws = torch.from_numpy(np.load(os.path.join(
            latents_dir, fname))).type(torch.FloatTensor).unsqueeze(0).cuda()
        attributes = torch.from_numpy(np.load(os.path.join(
            attributes_dir, fname))).type(
            # torch.FloatTensor).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
            torch.FloatTensor).unsqueeze(0).unsqueeze(-1).cuda()

        gender = attributes[0][1][0]
        # print(id, gender)
        assert gender == 0. or gender == 1.
        new_gender = 1. if gender == 0. else 0.
        new_attributes = attributes.clone()
        new_attributes[0][1][0] = new_gender

        # print('ws', ws, ws.shape)
        # print('attributes', attributes, attributes.shape)
        # print('new_attributes', new_attributes, new_attributes.shape)

        # z, _ = A(ws, attributes, zero_pad)
        # new_ws, _ = A(z, new_attributes, zero_pad, reverse=True)
        # new_ws[0][8:] = ws[0][8:]

        img = G.g(ws)[0]
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()

        # n_imgs = 2 + len(UNALTERED_WS_FROM_IDXS)
        n_imgs = 2 + len(VERSIONS)
        canvas = Image.new('RGB', (IMSIZE * n_imgs, IMSIZE), 'black')

        # original
        og = Image.open(OG_IMG_PATH.format(id))
        canvas.paste(og, (0, 0))

        # projected
        canvas.paste(
            Image.fromarray(
                np.transpose(img, (1, 2, 0)),
                'RGB',
            ),
            (IMSIZE, 0),
        )

        # swaps
        # for i, from_idx in enumerate(UNALTERED_WS_FROM_IDXS):
        for i, version in enumerate(VERSIONS):
            # if from_idx == NUM_WS:
            #     new_img = G.g(new_ws)[0]
            # else:
            #     mix_ws = new_ws.clone()
            #     mix_ws[0][from_idx:] = ws[0][from_idx:]
            #     new_img = G.g(mix_ws)[0]
            A = As[version]
            z, _ = A(ws, attributes, zero_pad)
            new_ws, _ = A(z, new_attributes, zero_pad, reverse=True)
            new_img = G.g(new_ws)[0]
            new_img = (new_img * 127.5 + 128).clamp(0, 255).to(
                torch.uint8).cpu().numpy()
            canvas.paste(
                Image.fromarray(
                    np.transpose(new_img, (1, 2, 0)),
                    'RGB',
                ),
                (IMSIZE * (2 + i), 0),
            )

        canvas.save(os.path.join(save_dir, f'{id}.png'))


if __name__ == '__main__':
    run()
