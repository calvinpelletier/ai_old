#!/usr/bin/env python3
import numpy as np
import torch
import clip
from PIL import Image
import copy
from ai_old.util.styleclip.manipulate import Manipulator
import argparse


def get_img_f(out, model, preprocess):
    imgs = out.reshape([-1] + list(out.shape[2:]))

    tmp = []
    for i in range(len(imgs)):
        img = Image.fromarray(imgs[i])
        image = preprocess(img).unsqueeze(0).to(device)
        tmp.append(image)

    image = torch.cat(tmp)
    with torch.no_grad():
        image_features = model.encode_image(image)
    image_features = image_features.cpu().numpy()
    image_features = image_features.reshape(list(out.shape[:2]) + [512])
    return image_features


def get_fs(fs):
    tmp = np.linalg.norm(fs, axis=-1)
    fs1 = fs / tmp[:, :, :, None]
    fs2 = fs1[:, :, 1, :] - fs1[:, :, 0, :]  # 5*sigma - (-5)* sigma
    fs3 = fs2 / np.linalg.norm(fs2, axis=-1)[:, :, None]
    fs3 = fs3.mean(axis=1)
    fs3 = fs3 / np.linalg.norm(fs3, axis=-1)[:, None]
    return fs3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ffhq')
    args = parser.parse_args()
    dataset_name = args.dataset_name

    device = 'cuda'
    # model, preprocess = clip.load('ViT-B/32', device=device)
    model, preprocess = clip.load('ViT-B/32', device=device, jit=False)

    M = Manipulator(dataset_name=dataset_name)
    np.set_printoptions(suppress=True)
    print(M.dataset_name)

    img_sindex = 0
    num_images = 100
    dlatents_o = []
    tmp = img_sindex * num_images
    for i in range(len(M.stylespace)):
        tmp1 = M.stylespace[i][tmp:(tmp + num_images)]
        dlatents_o.append(tmp1)

    path = f'/home/asiu/data/styleclip/npy/{dataset_name}/w.npy'
    ws = torch.tensor(np.load(path)[:num_images]).to('cuda')

    all_f = []
    M.alpha = [-5,5] # ffhq 5
    M.step = 2
    M.num_images = num_images
    # select = np.array(M.mindexs) <= 16 # below or equal to 128 resolution
    select = np.array(M.mindexs) <= 10 # below or equal to 128 resolution
    mindexs2 = np.array(M.mindexs)[select]
    for lindex in mindexs2: # ignore ToRGB layers
        print(lindex)
        num_c = M.stylespace[lindex].shape[1]
        for cindex in range(num_c):
            print(cindex)
            M.stylespace = copy.copy(dlatents_o)
            M.stylespace[lindex][:, cindex] = M.code_mean[lindex][cindex]

            M.manipulate_layers = [lindex]
            codes,out = M.edit_one_c(ws, cindex)
            image_features1 = get_img_f(out, model, preprocess)
            all_f.append(image_features1)

    all_f = np.array(all_f)
    fs3 = get_fs(all_f)

    path = f'/home/asiu/data/styleclip/npy/{M.dataset_name}/fs3.npy'
    np.save(path, fs3)
