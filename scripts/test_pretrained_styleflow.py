#!/usr/bin/env python3
import os
from PIL import Image
import torch
import numpy as np
from ai_old.util.factory import build_model_from_exp
from tqdm import tqdm
from external.styleflow.flow import cnf
import ai_old.constants as c
import pickle
import torch.nn.functional as F


NUM_WS = 18
NUM_ATTRS = 17
IMSIZE = 256


def generate(model, ws):
    img = model.synthesis(ws)
    img = F.interpolate(
        img,
        size=(IMSIZE, IMSIZE),
        mode='bilinear',
        align_corners=False,
    )
    return img


def run():
    with open('/home/asiu/data/models/stylegan/ffhq.pkl', 'rb') as f:
        G = pickle.load(f)['G_ema']
    G = G.eval().cuda()

    A = cnf(512, '512-512-512-512-512', NUM_ATTRS, 1)
    # A = cnf(512, '512-512-512-512-512-512-512', NUM_ATTRS, 1)
    A.load_state_dict(
        torch.load('/home/asiu/code/styleflow/flow_weight/modellarge10k.pt'))
    A = A.eval().cuda()

    all_latents = np.load('/home/asiu/data/styleflow/data_numpy/latents.npy')
    tmp_lighting = np.load('/home/asiu/data/styleflow/data_numpy/lighting.npy')
    tmp_attr = np.load('/home/asiu/data/styleflow/data_numpy/attributes.npy')
    all_attr = np.concatenate([tmp_lighting, tmp_attr], axis=1)
    all_latents = torch.from_numpy(all_latents).type(torch.FloatTensor).cuda()
    all_attr = torch.from_numpy(all_attr).type(torch.FloatTensor).cuda()
    print(all_latents.shape)
    print(all_attr.shape)
    save_dir = '/home/asiu/datasets/facegen-9-1/debug/swap'

    zero_pad = torch.zeros(1, NUM_WS, 1).cuda()

    for idx in tqdm(range(latents.shape[0])):
        id = f'{i:05d}'
        ws = latents[idx].unsqueeze(0)
        print(ws.shape)
        attributes = all_attr[idx].unsqueeze(0)
        print(ws.shape)

        gender = attributes[0][9][0]
        # print(id, gender)
        assert gender == 0. or gender == 1.
        new_gender = 1. if gender == 0. else 0.
        new_attributes = attributes.clone()
        new_attributes[0][9][0] = new_gender

        # print('ws', ws, ws.shape)
        # print('attributes', attributes, attributes.shape)
        # print('new_attributes', new_attributes, new_attributes.shape)

        z, _ = A(ws, attributes, zero_pad)
        new_ws, _ = A(z, new_attributes, zero_pad, reverse=True)
        # new_ws[0][8:] = ws[0][8:]

        img = generate(G, ws)[0]
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()

        n_imgs = 1 + (NUM_WS + 1) - 5
        canvas = Image.new('RGB', (IMSIZE * n_imgs, IMSIZE), 'black')
        canvas.paste(
            Image.fromarray(
                np.transpose(img, (1, 2, 0)),
                'RGB',
            ),
            (0, 0),
        )

        for i in range(5, NUM_WS + 1):
            if i == NUM_WS:
                new_img = G.g(new_ws)[0]
            else:
                mix_ws = new_ws.clone()
                mix_ws[0][i:] = ws[0][i:]
                new_img = generate(G, mix_ws)[0]
            new_img = (new_img * 127.5 + 128).clamp(0, 255).to(
                torch.uint8).cpu().numpy()
            canvas.paste(
                Image.fromarray(
                    np.transpose(new_img, (1, 2, 0)),
                    'RGB',
                ),
                (IMSIZE * (i - 4), 0),
            )

        canvas.save(os.path.join(save_dir, f'{id}.png'))


if __name__ == '__main__':
    run()
