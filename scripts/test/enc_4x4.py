#!/usr/bin/env python3
import torch
import ai_old.constants as c
from ai_old.util import config
from ai_old.util.factory import build_model_from_exp
from ai_old.util.factory import build_dataset
from tqdm import tqdm
import os


config_path = os.path.join(c.CONFIGS_FOLDER, 'train/enc-lerp/1/0.yaml')
cfg = config.load(config_path)

lerp_and_gen = build_model_from_exp(
    'lerp/5/5',
    'G',
    return_cfg=False,
).eval().requires_grad_(False).to('cuda')

ae = build_model_from_exp(
    'rec/25/8',
    'G_ema',
    return_cfg=False,
).eval().requires_grad_(False).to('cuda')

ae2 = build_model_from_exp(
    'rec/25/8',
    'G_ema',
    return_cfg=False,
).eval().requires_grad_(False).to('cuda')

batch_size = 32
dataset_core = build_dataset(cfg.dataset)
val_set = dataset_core.get_val_set(
    batch_size,
    0, # seed
    0, # rank
    1, # num gpus
    verbose=False,
)

with torch.no_grad():
    for batch in tqdm(val_set):
        img = batch['img'].to('cuda').to(torch.float32) / 127.5 - 1
        gender = batch['gender'].to('cuda').to(
            torch.float32).unsqueeze(1)
        w = batch['w'].to('cuda').to(torch.float32)

        base_enc = ae.e(img)
        base_enc2 = ae2.e(img)

        tmp_path = '/home/asiu/data/tmp/debug.pt'
        torch.save(base_enc, tmp_path)
        base_enc3 = torch.load(tmp_path).to('cuda').to(torch.float32)

        base_enc4 = batch['base_enc'].to('cuda').to(torch.float32)

        print('base_enc', base_enc.shape, base_enc[0, 0, 0])
        print('base_enc2', base_enc2.shape, base_enc2[0, 0, 0])
        print('base_enc3', base_enc3.shape, base_enc3[0, 0, 0])
        print('base_enc4', base_enc4.shape, base_enc4[0, 0, 0])
        assert (base_enc == base_enc2).all()
        assert (base_enc == base_enc3).all()
        assert (base_enc == base_enc4).all()
