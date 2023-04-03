#!/usr/bin/env python3
from external.e4e.models.stylegan2.model import Generator
import os
import ai_old.constants as c
import torch
import pickle
from ai_old.dataset import DatasetBase
import ai_old.dataset.filter_func as ff
from PIL import Image
import numpy as np
from time import time
from external.sg2.unit import Generator as Generator3


ckpt = torch.load(os.path.join(
    c.PRETRAINED_MODELS,
    'stylegan/stylegan2-ffhq-config-f.pt',
))
g1 = Generator(1024, 512, 8).to('cuda').eval()
g1.load_state_dict(ckpt['g_ema'], strict=False)
g1.requires_grad_(False)


with open('/home/asiu/data/models/stylegan/ffhq.pkl', 'rb') as f:
    g2 = pickle.load(f)['G_ema']
g2 = g2.eval().to('cuda')
# torch.save(g2.state_dict(), '/home/asiu/data/models/stylegan/official_g.pt')

g3 = Generator3(
    512, # z dims
    0, # c dims
    512, # w dims
    1024, # imsize
    3, # channels
    mapping_kwargs={
        'num_layers': 8,
    },
    synthesis_kwargs={
        'channel_base': 32768,
        'channel_max': 512,
        'num_fp16_res': 4,
        'conv_clamp': 256,
    },
)
g3.load_state_dict(torch.load(
    '/home/asiu/data/models/stylegan/official_g.pt'))
g3 = g3.to('cuda').eval()


class DS(DatasetBase):
    def filter_func(self):
        return ff.for_dataset('ffhq-128')

    def select_cols(self):
        return {
            'item_id': 'id',
            'e4e_inv_w_plus': 'w_plus',
        }
dataset = DS(False).inference(batch_size=8, seed=0, rank=0, num_gpus=1)

with torch.no_grad():
    for batch in dataset:
        w_plus = batch['w_plus'].to('cuda').to(torch.float32)
        imgs1 = g1([w_plus], input_is_latent=True)[0]
        imgs2 = g2.synthesis(w_plus)
        imgs3 = g3.synthesis(w_plus)

        imgs1 = (imgs1 * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        imgs2 = (imgs2 * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        imgs3 = (imgs3 * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()

        for id, img1, img2, img3 in zip(batch['id'], imgs1, imgs2, imgs3):
            print(id)
            canvas = Image.new(
                'RGB',
                (1024 * 3, 1024),
                'black',
            )
            for j, img in enumerate([img1, img2, img3]):
                canvas.paste(
                    Image.fromarray(
                        np.transpose(img, (1, 2, 0)),
                        'RGB',
                    ),
                    (1024 * j, 0),
                )
            canvas.save(f'/home/asiu/data/tmp/asdf3/{id}.png')
