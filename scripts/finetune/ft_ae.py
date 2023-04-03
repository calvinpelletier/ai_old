#!/usr/bin/env python3
import torch
import os
from ai_old.trainer.pti import PtiTrainer
from PIL import Image
from ai_old.util.face import align
import numpy as np
from ai_old.util.factory import build_model_from_exp
from ai_old.util.etc import resize_imgs, normalized_tensor_to_pil_img, \
    create_img_row, create_img_grid, pil_to_tensor
from ai_old.util.pretrained import build_pretrained_e4e, build_pretrained_sg2
import matplotlib.pyplot as plt
from ai_old.finetune.ae import finetune_ae


FOLDER = '/home/asiu/data/tmp/ft-ae'
W_LERP_EXP = 'lerp/5/5'
AE_EXP = 'rec/25/8'
OPT = 'adam'
LR = 0.4
MAX_MAG = 1.5
BATCH_SIZE = 8
N_ITER = 100


def run(id, full_path, face_idx, gender):
    dir = os.path.join(FOLDER, id)
    os.makedirs(dir, exist_ok=True)

    gender = torch.tensor(gender).to('cuda').unsqueeze(0)

    # crop and align
    print('crop and align')
    full_img, aligned, inner_quad = align(full_path, face_idx)
    aligned_256 = aligned.resize((256, 256), Image.LANCZOS)
    img_tensor = pil_to_tensor(aligned)
    img_tensor_256 = pil_to_tensor(aligned_256)

    # invert
    w_path = os.path.join(dir, 'w.npy')
    if os.path.isfile(w_path):
        w = torch.from_numpy(np.load(w_path)).to('cuda').to(torch.float32)
    else:
        print('e4e inversion')
        E = build_pretrained_e4e()
        G = build_pretrained_sg2().synthesis
        with torch.no_grad():
            w = E(img_tensor_256)
        np.save(w_path, w.cpu().numpy())
        del E
        del G

    # pti
    pti_model_path = os.path.join(dir, 'pti_model.pt')
    if os.path.isfile(pti_model_path):
        pti_G = build_pretrained_sg2(path_override=pti_model_path)
    else:
        print('finetune generator')
        pti_trainer = PtiTrainer('cuda')
        pti_G = pti_trainer.train(img_tensor, w)
        torch.save(pti_G.state_dict(), pti_model_path)
    pti_G = pti_G.synthesis
    with torch.no_grad():
        pti_inverted_tensor = resize_imgs(pti_G(w), 256)
        pti_inverted = normalized_tensor_to_pil_img(pti_inverted_tensor[0])

    # models
    w_lerper = build_model_from_exp(
        W_LERP_EXP, 'G', return_cfg=False).f.eval().to('cuda')
    ae = build_model_from_exp(
        AE_EXP, 'G_ema', return_cfg=False).eval().to('cuda')

    # prep for multi-mag batches
    w = w.repeat(BATCH_SIZE, 1, 1)
    gender = gender.repeat(BATCH_SIZE, 1)
    mags = torch.tensor(np.linspace(0., MAX_MAG, num=BATCH_SIZE), device='cuda')

    with torch.no_grad():
        # calc target
        target = resize_imgs(pti_G(w_lerper(w, gender, magnitude=mags)), 256)

        # samples
        gt_samples = [normalized_tensor_to_pil_img(x) for x in target]
        ae_samples = [normalized_tensor_to_pil_img(x) for x in ae(
            target,
            noise_mode='const',
        )]

    # finetune
    ae_lc, losses = finetune_ae(
        ae,
        target,
        opt_type=OPT,
        lr=LR,
        n_iter=N_ITER,
        return_losses=True,
    )

    # loss graph
    plt.plot(losses)
    plt.savefig(os.path.join(dir, 'loss.png'))
    plt.clf()

    # samples
    with torch.no_grad():
        enc = ae_lc()
        ft_samples = [normalized_tensor_to_pil_img(x) for x in ae.g(
            enc,
            noise_mode='const',
        )]
    create_img_grid([gt_samples, ae_samples, ft_samples], 256).save(
        os.path.join(dir, 'samples.png'))


if __name__ == '__main__':
    run('sera13', '/home/asiu/data/sera/og/13.jpg', 1, 1.)

    # lafleur_path = '/home/asiu/data/tmp/e2e/lafleur-small/og_full.jpg'
    # run('lafleur-small', lafleur_path, 0, 1.)
    #
    # lafleur_path = '/home/asiu/data/tmp/e2e/lafleur-large/og_full.jpg'
    # run('lafleur-large', lafleur_path, 0, 1.)
    #
    # run('sera12', '/home/asiu/data/sera/og/12.jpg', 0, 1.)
