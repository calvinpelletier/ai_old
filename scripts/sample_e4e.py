#!/usr/bin/env python3
import os
import ai_old.constants as c
from ai_old.util.etc import make_deterministic
from PIL import Image
from ai_old.util.factory import build_model_from_exp
import torch
import numpy as np
from tqdm import tqdm


def run():
    make_deterministic()

    facegen_exp = 'facegen/9/1'
    base_dir = os.path.join(
        c.ASI_DATASETS_PATH,
        facegen_exp.replace('/', '-'),
    )
    img_dir = os.path.join(base_dir, 'imgs')
    latent_dir = os.path.join(base_dir, 'latents')
    device = torch.device('cuda', 0)
    batch_size = 32
    n_samples_approx = 10000
    n_batches = n_samples_approx // batch_size
    truncation = 0.7

    # load from exp
    model, _ = build_model_from_exp(facegen_exp, 'G_ema')
    model = model.eval().requires_grad_(False).to(device)

    # loop
    for i in tqdm(range(n_batches)):
        z = torch.randn([batch_size, model.z_dims], device=device)
        ws = model.f(z, None, truncation_psi=truncation, skip_w_avg_update=True)
        imgs = model.g(ws)
        imgs = (imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        ws = ws.cpu().numpy()
        for j in range(batch_size):
            id = f'{i*batch_size+j:05d}'
            np.save(os.path.join(latent_dir, f'{id}.npy'), ws[j])
            Image.fromarray(
                np.transpose(imgs[j], (1, 2, 0)),
                'RGB',
            ).save(os.path.join(img_dir, f'{id}.png'))


if __name__ == '__main__':
    run()
