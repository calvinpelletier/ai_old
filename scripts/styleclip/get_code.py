#!/usr/bin/env python3
from external.e4e.models.stylegan2.model import Generator
import os
import pickle
import numpy as np
import argparse
import torch
from ai_old.util.etc import make_deterministic
import tqdm


def load_model():
    path = '/home/asiu/data/models/stylegan/stylegan2-ffhq-config-f.pt'
    ckpt = torch.load(path)
    model = Generator(1024, 512, 8).to('cuda')
    model.load_state_dict(ckpt['g_ema'], strict=False)
    model = model.eval()
    latent_avg = ckpt['latent_avg']
    return model, latent_avg


def lerp(a, b, t):
     return a + (b - a) * t


def get_code(G, dlatent_avg, num_img, num_once, dataset_name):
    truncation_psi = 0.7
    truncation_cutoff = 8

    dlatent_avg = dlatent_avg.numpy()

    dlatents = np.zeros((num_img, 512), dtype='float32')
    for i in range(int(num_img / num_once)):
        src_latents = torch.randn(
            [num_once, 512],
            device='cuda',
        )

        src_dlatents = G.style(src_latents).unsqueeze(dim=1).repeat(
            1, G.n_latent, 1).cpu().detach().numpy()

        # truncation
        layer_idx = np.arange(src_dlatents.shape[1])[np.newaxis, :, np.newaxis]
        ones = np.ones(layer_idx.shape, dtype=np.float32)
        coefs = np.where(
            layer_idx < truncation_cutoff,
            truncation_psi * ones,
            ones,
        )
        # print(dlatent_avg.shape)
        # print(src_dlatents.shape)
        # print(coefs.shape)
        # print(coefs)
        src_dlatents_np = lerp(dlatent_avg, src_dlatents, coefs)
        src_dlatents = src_dlatents_np[:, 0, :].astype('float32')
        dlatents[(i*num_once):((i+1)*num_once), :] = src_dlatents

    np.save(f'/home/asiu/data/styleclip/npy/{dataset_name}/w.npy', dlatents)


def _append_or_concat(arr, idx, val):
    if idx >= len(arr):
        arr.append(val)
    else:
        arr[idx] = torch.cat([arr[idx], val], dim=0)


def get_s(dataset_name, num_img):
    path = f'/home/asiu/data/styleclip/npy/{dataset_name}/w.npy'
    dlatents = torch.tensor(np.load(path)[:num_img]).to('cuda')

    G, _ = load_model()

    layer_names = [
        '0_conv1',
        # '0_rbg',
    ]
    for i in range(len(G.convs[::2])):
        layer_names.append(f'{i+1}_conv1')
        layer_names.append(f'{i+1}_conv2')
        # layer_names.append(f'{i+1}_rgb')
    # print(layer_names)

    stylespace = []
    for w in dlatents:
        w = w.unsqueeze(0).unsqueeze(1).repeat(1, G.n_latent, 1)

        idx = 0
        _append_or_concat(stylespace, idx, G.conv1.conv.modulation(w[:, 0]))
        idx += 1
        # _append_or_concat(stylespace, idx, G.to_rgb1.conv.modulation(w[:, 1]))
        # idx += 1

        i = 1
        for conv1, conv2, rgb in zip(G.convs[::2], G.convs[1::2], G.to_rgbs):
            _append_or_concat(stylespace, idx, conv1.conv.modulation(w[:, i]))
            idx += 1
            _append_or_concat(stylespace, idx, conv2.conv.modulation(w[:, i+1]))
            idx += 1
            # _append_or_concat(stylespace, idx, rgb.conv.modulation(w[:, i+2]))
            # idx += 1
            i += 2

    # print(len(stylespace))
    # for i, x in enumerate(stylespace):
    #     print(layer_names[i], x.shape)

    for i in range(len(stylespace)):
        stylespace[i] = stylespace[i].cpu().detach().numpy()

    return layer_names, stylespace


def get_code_ms(stylespace):
        means = []
        stds = []
        for s in stylespace:
            means.append(s.mean(axis=0))
            stds.append(s.std(axis=0))
        return means, stds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ffhq')
    parser.add_argument('--code_type',
        choices=['w', 's', 's_mean_std'],
        default='w',
    )

    args = parser.parse_args()
    num_img = 100000
    num_once = 1000
    dataset_name = args.dataset_name

    make_deterministic()

    if args.code_type == 'w':
        G, dlatent_avg = load_model()
        get_code(G, dlatent_avg, num_img, num_once, dataset_name)

    elif args.code_type == 's':
        s = get_s(dataset_name, num_img=2000)
        path = f'/home/asiu/data/styleclip/npy/{dataset_name}/s.pkl'
        with open(path, 'wb') as f:
            pickle.dump(s, f)

    elif args.code_type == 's_mean_std':
        stylespace = get_s(dataset_name, num_img=num_img)[1]
        ms = get_code_ms(stylespace)
        path = f'/home/asiu/data/styleclip/npy/{dataset_name}/s_mean_std.pkl'
        with open(path, 'wb') as f:
            pickle.dump(ms, f)
