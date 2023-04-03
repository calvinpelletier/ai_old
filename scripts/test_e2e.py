#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import os
from ai_old.trainer.pti import PtiTrainer
from PIL import Image
from ai_old.util.face import get_faces, get_landmarks, align_face, \
    custom_align_face
import numpy as np
from ai_old.util.factory import build_model_from_exp
from ai_old.util.etc import resize_imgs, normalized_tensor_to_pil_img, \
    create_img_row, create_img_grid, AttrDict
from ai_old.nn.models.lerp.static import convert_dynamic_lerper_to_static
from ai_old.loss.lerp import SoloClassifyLerpLoss
from ai_old.util.inverse import get_outer_quad, solo_aligned_to_fam_final
from ai_old.util.pretrained import build_pretrained_e4e, build_pretrained_sg2
from ai_old.util.outer import get_outer_boundary_mask, get_dilate_kernel, \
    get_inner_mask
from ai_old.nn.models.seg import Segmenter, colorize, binary_seg_to_img
import cv2
from tqdm import tqdm
from ai_old.nn.models.inpaint.aot import AotInpainter
import matplotlib.pyplot as plt
from lpips import LPIPS
from random import random


FOLDER = '/home/asiu/data/tmp/e2e'
MAGS = [0., 0.5, 0.75, 1., 1.25]
GROUPS = [
    ['skin', 'nose', 'glasses', 'l_eye', 'r_eye', 'l_brow', 'r_brow',
        'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'earring', 'neck'],
    ['hair', 'hat'],
    ['bg'],
    ['necklace', 'cloth'],
]


def align(full_path, face_idx):
    img = Image.open(full_path)
    img_np = np.asarray(img)
    box = get_faces(img_np)[face_idx]
    landmarks = get_landmarks(img_np, box)
    aligned, inner_quad = align_face(img, landmarks, 1024)
    return img, aligned, inner_quad


def outer_align(full_img, inner_quad):
    outer_quad = get_outer_quad(inner_quad, full=full_img)
    outer_imsize = 1024 + 512
    outer_aligned = custom_align_face(full_img, outer_quad, outer_imsize)
    return outer_aligned, outer_quad


def get_lerper(exp):
    lerper = build_model_from_exp(exp, 'G', return_cfg=False).f
    lerper = lerper.to('cuda')
    lerper.eval()
    return lerper


def run_G(G, w, return_tensor=False):
    tensor = resize_imgs(G(w), 256)
    img = normalized_tensor_to_pil_img(tensor[0])
    if return_tensor:
        return tensor, img
    return img


def get_and_run_lerper(exp, mags, G, w, gender):
    lerper = get_lerper(exp)
    swaps = run_lerper(lerper, mags, G, w, gender)
    return lerper, swaps


def run_lerper(lerper, mags, G, w, gender):
    swaps = []
    for mag in mags:
        swap_w = lerper(w, gender, magnitude=mag)
        swap = run_G(G, swap_w)
        swaps.append(swap)
    return swaps


def finetune_lerper(G, lerper, img, w, gender):
    ft_lerper = convert_dynamic_lerper_to_static(
        lerper, w, gender, type='levels')
    ft_lerper = ft_lerper.to('cuda').train()

    opt = torch.optim.Adam(ft_lerper.parameters(), lr=0.002)

    loss_fn = SoloClassifyLerpLoss(
        img,
        w,
        gender,
        imsize=256,
        face_weight=0.1,
        delta_weight=0.8,
        classify_weight=1.,
        use_l2_for_classify=False,
    ).to('cuda')

    for i in tqdm(range(100)):
        opt.zero_grad()
        new_w = ft_lerper(w, gender, magnitude=1.)
        new_img = G(new_w, noise_mode='const')
        new_img = resize_imgs(new_img, 256)
        loss = loss_fn(new_img, new_w)
        loss.backward()
        opt.step()

    return ft_lerper


def pil_to_tensor(img):
    img_np = np.asarray(img).transpose(2, 0, 1)
    img_tensor = torch.from_numpy(np.copy(img_np))
    img_tensor = img_tensor.to('cuda').to(torch.float32) / 127.5 - 1
    return img_tensor.unsqueeze(0)


def get_outer_seg_predictor():
    outer_seg_predictor = build_model_from_exp(
        'outer-seg/0/1',
        'model',
        return_cfg=False,
    ).to('cuda')
    outer_seg_predictor.eval()
    assert not outer_seg_predictor.pred_from_seg_only
    return outer_seg_predictor


def resize_and_pad_inner_img(
    img,
    inner_mask,
    inner_imsize,
    outer_imsize,
    is_seg=False,
):
    if is_seg:
        assert img.shape == (1, inner_imsize, inner_imsize)
        ret = torch.zeros(1, outer_imsize, outer_imsize, dtype=torch.long,
            device='cuda')
    else:
        img = resize_imgs(img, inner_imsize)
        ret = torch.zeros(1, 3, outer_imsize, outer_imsize, device='cuda')
    buf = (outer_imsize - inner_imsize) // 2
    for y in range(outer_imsize):
        for x in range(outer_imsize):
            is_inner = y >= buf and y < (buf + inner_imsize) and \
                x >= buf and x < (buf + inner_imsize)
            if is_inner:
                assert inner_mask[y][x] == 1.
                if is_seg:
                    ret[0, y, x] = img[0, y - buf, x - buf]
                else:
                    ret[0, :, y, x] = img[0, :, y - buf, x - buf]
            else:
                assert inner_mask[y][x] == 0.
    return ret


def dilate_mask(mask, dilate_kernel):
    dilated_mask = cv2.dilate(mask.cpu().numpy() * 255., dilate_kernel)
    dilated_mask = torch.tensor(dilated_mask / 255.).to('cuda')
    return (dilated_mask > 0.5).float()


def fhbc_seg_to_facehair(seg):
    return torch.bitwise_or(
        seg == 0, # face
        seg == 1, # hair
    ).float()


def get_mask_combo(inner_gan_mask, inpaint_mask, gt_mask):
    combo = torch.zeros_like(gt_mask).to(torch.long)
    h, w = combo.shape
    for y in range(h):
        for x in range(w):
            assert (inner_gan_mask[y][x] + inpaint_mask[y][x] + \
                gt_mask[y][x]) == 1.
            if inner_gan_mask[y][x] == 1.:
                combo[y][x] = 1
            elif inpaint_mask[y][x] == 1.:
                combo[y][x] = 2
            else:
                pass # ... = 0
    return combo


def get_inpainter(aot=False):
    if aot:
        inpainter = AotInpainter(AttrDict(
            {'dataset': AttrDict({'imsize': 512})}))
        path = '/home/asiu/data/models/aot/G.pt'
        inpainter.load_state_dict(torch.load(path, map_location='cuda'))
    else:
        inpainter = build_model_from_exp(
            'outpaint/1/3', 'G_ema', return_cfg=False)
    return inpainter.to('cuda').eval()


def seg_and_inpaint(swap_tensor, outer_tensor, aot=False):
    if aot:
        inner_imsize = int(512 / 1.5)
        outer_imsize = 512
    else:
        inner_imsize = 128
        outer_imsize = 192

    # segment
    print('segmentation')
    segmenter = Segmenter().to('cuda')
    segmenter.eval()
    swap_seg = torch.argmax(segmenter(
        swap_tensor, groups=GROUPS, output_imsize=128)[0], dim=0)
    outer_seg = torch.argmax(segmenter(
        outer_tensor, groups=GROUPS, output_imsize=outer_imsize)[0], dim=0)

    # predict swap outer seg from swap inner seg
    print('outer seg prediction')
    outer_seg_predictor = get_outer_seg_predictor()
    swap_outer_seg = outer_seg_predictor(
        resize_and_pad_inner_img(
            swap_seg.unsqueeze(0),
            outer_seg_predictor.inner_mask,
            128,
            192,
            is_seg=True,
        ),
        resize_and_pad_inner_img(
            swap_tensor,
            outer_seg_predictor.inner_mask,
            128,
            192,
        ),
    )
    swap_outer_seg = resize_imgs(swap_outer_seg, outer_imsize)
    swap_outer_seg = torch.argmax(swap_outer_seg[0], dim=0)

    # calc masks
    print('mask calc')
    facehair = fhbc_seg_to_facehair(outer_seg)
    swap_facehair = fhbc_seg_to_facehair(swap_outer_seg)
    inner_mask = get_inner_mask(inner_imsize, outer_imsize)
    inner_gan_mask = swap_facehair * inner_mask
    inv_inner_gan_mask = 1. - inner_gan_mask
    dilate_kernel = get_dilate_kernel(outer_imsize)
    dilated_facehair = dilate_mask(facehair, dilate_kernel)
    dilated_swap_facehair = dilate_mask(swap_facehair, dilate_kernel)
    outer_boundary_mask = get_outer_boundary_mask(outer_imsize).to('cuda')
    inv_outer_boundary_mask = 1. - outer_boundary_mask
    dilated_facehair_union = torch.clamp(
        dilated_facehair + dilated_swap_facehair, min=0., max=1.)
    inpaint_mask = dilated_facehair_union * inv_inner_gan_mask * \
        inv_outer_boundary_mask
    gt_mask = torch.ones_like(inpaint_mask) * (1. - inpaint_mask) * \
        inv_inner_gan_mask

    # merge and inpaint
    print('merge and inpaint')
    padded_swap = resize_and_pad_inner_img(
        swap_tensor, inner_mask, inner_imsize, outer_imsize)
    outer_tensor = resize_imgs(outer_tensor, outer_imsize)
    merged = outer_tensor * gt_mask + padded_swap * inner_gan_mask
    inpainter = get_inpainter(aot=aot)
    inpainted = inpainter(merged, inpaint_mask.unsqueeze(0))[0]
    inpainted = normalized_tensor_to_pil_img(inpainted)

    # debug
    naive_merge = outer_tensor * inv_inner_gan_mask + \
        padded_swap * inner_gan_mask
    naive_merge = normalized_tensor_to_pil_img(naive_merge[0])
    merged = normalized_tensor_to_pil_img(merged[0])

    return naive_merge, merged, inpainted


def finetune_student(teacher, lerper, w, is_mtf):
    teacher.requires_grad_(False)
    teacher.eval()

    lerper.requires_grad_(False)
    lerper.eval()

    student = build_model_from_exp('distill/1/1', 'G_ema', return_cfg=False)
    student = student.to('cuda')
    student.requires_grad_(True)
    student.train()

    opt = torch.optim.Adam(student.parameters(), lr=3e-4)
    lpips = LPIPS(net='alex').to('cuda').eval()

    debug_data = []
    samples = []
    n_iter = 400
    for i in tqdm(range(n_iter)):
        opt.zero_grad()

        new_w = lerper(w, is_mtf, magnitude=random() * 1.5)
        teacher_img = teacher(new_w, noise_mode='const')
        teacher_img = resize_imgs(teacher_img, 256)
        student_img = student(new_w.detach())

        lpips_loss = lpips(student_img, teacher_img.detach()).mean()
        pixel_loss = F.mse_loss(student_img, teacher_img.detach())
        loss = lpips_loss + pixel_loss
        early_stop = loss.item() < 0.03

        debug_data.append(loss.item())
        if i % 50 == 0 or i == n_iter - 1 or early_stop:
            row = []
            for mag in [0., 0.5, 1., 1.5]:
                row.append(normalized_tensor_to_pil_img(student(
                    lerper(w, is_mtf, magnitude=mag),
                )[0]))
            samples.append(row)

        loss.backward()
        opt.step()

        if early_stop:
            break

    row = []
    for mag in [0., 0.5, 1., 1.5]:
        row.append(normalized_tensor_to_pil_img(resize_imgs(teacher(
            lerper(w, is_mtf, magnitude=mag),
            noise_mode='const',
        ), 256)[0]))
    samples.append(row)

    return student, samples, debug_data


def finetune_student_adv(teacher, lerper, w, is_mtf):
    teacher.requires_grad_(False)
    teacher.eval()

    lerper.requires_grad_(False)
    lerper.eval()

    student = build_model_from_exp('distill/1/1', 'G_ema', return_cfg=False)
    student = student.to('cuda')
    student.requires_grad_(True)
    student.train()

    D = build_model_from_exp('distill/1/1', 'D', return_cfg=False)
    D = D.to('cuda')
    D.requires_grad_(True)
    D.train()

    opt_G = torch.optim.Adam(student.parameters(), lr=2e-4)
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4)
    lpips = LPIPS(net='alex').to('cuda').eval()

    rec_losses = []
    gan_losses = []
    samples = []
    n_iter = 400
    for i in tqdm(range(n_iter)):
        # generator
        opt_G.zero_grad()
        student.requires_grad_(True)
        D.requires_grad_(False)

        new_w = lerper(w, is_mtf, magnitude=random() * 1.5)
        teacher_img = teacher(new_w, noise_mode='const')
        teacher_img = resize_imgs(teacher_img, 256)
        student_img = student(new_w.detach())

        gan_loss = F.softplus(-D(student_img))
        lpips_loss = lpips(student_img, teacher_img.detach()).mean()
        pixel_loss = F.mse_loss(student_img, teacher_img.detach())
        rec_loss = lpips_loss + pixel_loss
        loss = gan_loss + 20. * rec_loss
        early_stop = rec_loss.item() < 0.01

        rec_losses.append(rec_loss.item())
        gan_losses.append(gan_loss.item())
        if i % 50 == 0 or i == n_iter - 1 or early_stop:
            row = []
            for mag in [0., 0.5, 1., 1.5]:
                row.append(normalized_tensor_to_pil_img(student(
                    lerper(w, is_mtf, magnitude=mag),
                )[0]))
            samples.append(row)

        loss.backward()
        opt_G.step()

        # discriminator
        opt_D.zero_grad()
        student.requires_grad_(False)
        D.requires_grad_(True)
        student_img = student(new_w.detach())
        loss = F.softplus(D(student_img)) + F.softplus(-D(teacher_img.detach()))
        loss = loss.mean()
        loss.backward()
        opt_D.step()

        if early_stop:
            break

    row = []
    for mag in [0., 0.5, 1., 1.5]:
        row.append(normalized_tensor_to_pil_img(resize_imgs(teacher(
            lerper(w, is_mtf, magnitude=mag),
            noise_mode='const',
        ), 256)[0]))
    samples.append(row)

    return student, samples, rec_losses, gan_losses


def run(id, full_path, face_idx, gender):
    dir = os.path.join(FOLDER, id)
    os.makedirs(dir, exist_ok=True)

    gender = torch.tensor(gender).to('cuda').unsqueeze(0)

    # crop and align
    print('crop and align')
    full_img, aligned, inner_quad = align(full_path, face_idx)
    aligned_256 = aligned.resize((256, 256), Image.LANCZOS)
    outer_aligned, outer_quad = outer_align(full_img, inner_quad)
    outer_aligned_192 = outer_aligned.resize((192, 192), Image.LANCZOS)

    # convert to tensor
    img_tensor = pil_to_tensor(aligned)
    img_tensor_256 = pil_to_tensor(aligned_256)
    outer_tensor = pil_to_tensor(outer_aligned)

    # invert
    print('e4e inversion')
    E = build_pretrained_e4e()
    G = build_pretrained_sg2().synthesis
    with torch.no_grad():
        e4e_w = E(img_tensor_256)
        e4e_inverted = run_G(G, e4e_w)

        # cluster swap
        cluster_lerper, cluster_swaps = get_and_run_lerper(
            'lerp/3/0', MAGS, G, e4e_w, gender)

        # classify swap
        classify_lerper, classify_swaps = get_and_run_lerper(
            'lerp/5/5', MAGS, G, e4e_w, gender)

    # pti
    pti_trainer = PtiTrainer('cuda')
    # print('opt inversion')
    # pti_w = pti_trainer.get_inversion(img_tensor)
    print('finetune generator')
    # pti_G = pti_trainer.train(img_tensor, pti_w).synthesis
    pti_G = pti_trainer.train(img_tensor, e4e_w).synthesis
    with torch.no_grad():
        # opt_inverted = run_G(G, pti_w)
        # pti_inverted_tensor, pti_inverted = run_G(
        #     pti_G, pti_w, return_tensor=True)
        pti_inverted_tensor, pti_inverted = run_G(
            pti_G, e4e_w, return_tensor=True)

    # save inversions
    row = [aligned_256, e4e_inverted, pti_inverted]
    create_img_row(row, 256).save(os.path.join(dir, 'invs.png'))

    # pti cluster swap
    print('cluster swap')
    pti_cluster_swaps = run_lerper(
        cluster_lerper, MAGS, pti_G, e4e_w, gender)

    # pti classify swap
    print('classify swap')
    pti_classify_swaps = run_lerper(
        classify_lerper, MAGS, pti_G, e4e_w, gender)

    # finetune classify lerper on pti model
    print('finetune lerper')
    ft_classify_lerper = finetune_lerper(
        pti_G, classify_lerper, pti_inverted_tensor, e4e_w, gender)

    # finetuned pti classify swap
    print('finetuned classify swap')
    ft_pti_classify_swaps = run_lerper(
        ft_classify_lerper, MAGS, pti_G, e4e_w, gender)

    # best_swap = ft_pti_classify_swaps[3]
    best_swap = pti_cluster_swaps[3]
    swap_tensor = pil_to_tensor(best_swap)

    # tmp
    w1 = cluster_lerper(e4e_w, gender, magnitude=1.)
    w2 = classify_lerper(e4e_w, gender, magnitude=1.)
    w3 = ft_classify_lerper(e4e_w, gender, magnitude=1.)
    for w_a in [w1, w2, w3]:
        for w_b in [w1, w2, w3]:
            print((w_a.mean(dim=1) - w_b.mean(dim=1)).norm(dim=1))

    # save swaps
    grid = [
        [aligned_256] + cluster_swaps,
        [aligned_256] + classify_swaps,
        [aligned_256] + pti_cluster_swaps,
        [aligned_256] + pti_classify_swaps,
        [aligned_256] + ft_pti_classify_swaps,
    ]
    create_img_grid(grid, 256).save(os.path.join(dir, 'swaps.png'))

    # # segment
    # print('segmentation')
    # segmenter = Segmenter().to('cuda')
    # segmenter.eval()
    # swap_seg = torch.argmax(segmenter(
    #     swap_tensor, groups=GROUPS, output_imsize=128)[0], dim=0)
    # outer_seg = torch.argmax(segmenter(
    #     outer_tensor, groups=GROUPS, output_imsize=192)[0], dim=0)
    #
    # # predict swap outer seg from swap inner seg
    # print('outer seg prediction')
    # outer_seg_predictor = get_outer_seg_predictor()
    # padded_swap_seg = resize_and_pad_inner_img(
    #     swap_seg.unsqueeze(0),
    #     outer_seg_predictor.inner_mask,
    #     128,
    #     192,
    #     is_seg=True,
    # )
    # padded_swap_img = resize_and_pad_inner_img(
    #     swap_tensor,
    #     outer_seg_predictor.inner_mask,
    #     128,
    #     192,
    # )
    # swap_outer_seg = outer_seg_predictor(padded_swap_seg, padded_swap_img)
    # swap_outer_seg = torch.argmax(swap_outer_seg[0], dim=0)
    #
    # # calc masks
    # print('mask calc')
    # facehair = fhbc_seg_to_facehair(outer_seg)
    # swap_facehair = fhbc_seg_to_facehair(swap_outer_seg)
    # inner_gan_mask = swap_facehair * outer_seg_predictor.inner_mask
    # inv_inner_gan_mask = 1. - inner_gan_mask
    # dilate_kernel = get_dilate_kernel(192)
    # dilated_facehair = dilate_mask(facehair, dilate_kernel)
    # dilated_swap_facehair = dilate_mask(swap_facehair, dilate_kernel)
    # outer_boundary_mask = get_outer_boundary_mask(192).to('cuda')
    # inv_outer_boundary_mask = 1. - outer_boundary_mask
    # dilated_facehair_union = torch.clamp(
    #     dilated_facehair + dilated_swap_facehair, min=0., max=1.)
    # inpaint_mask = dilated_facehair_union * inv_inner_gan_mask * \
    #     inv_outer_boundary_mask
    # gt_mask = torch.ones_like(inpaint_mask) * (1. - inpaint_mask) * \
    #     inv_inner_gan_mask
    #
    # # save segs and masks
    # print('visualize masks/segs')
    # row = [
    #     outer_aligned_192,
    #     colorize(outer_seg.unsqueeze(0), needs_argmax=False)[0],
    #     colorize(padded_swap_seg, needs_argmax=False)[0],
    #     colorize(swap_outer_seg.unsqueeze(0), needs_argmax=False)[0],
    # ]
    # create_img_row(row, 192).save(os.path.join(dir, 'segs.png'))
    # row = [
    #     outer_aligned_192,
    #     binary_seg_to_img(facehair),
    #     binary_seg_to_img(swap_facehair),
    #     binary_seg_to_img(dilated_facehair),
    #     binary_seg_to_img(dilated_swap_facehair),
    #     binary_seg_to_img(dilated_facehair_union),
    #     binary_seg_to_img(inner_gan_mask),
    #     binary_seg_to_img(inpaint_mask),
    #     binary_seg_to_img(gt_mask),
    #     colorize(get_mask_combo(
    #         inner_gan_mask,
    #         inpaint_mask,
    #         gt_mask,
    #     ).unsqueeze(0), needs_argmax=False)[0],
    # ]
    # create_img_row(row, 192).save(os.path.join(dir, 'masks.png'))
    #
    # # merge and inpaint
    # print('merge')
    # outer_tensor = resize_imgs(outer_tensor, 192)
    # merged = outer_tensor * gt_mask + padded_swap_img * inner_gan_mask
    # print('inpaint')
    # inpainter = get_inpainter()
    # inpainted = inpainter(merged, inpaint_mask.unsqueeze(0))[0]
    # inpainted = normalized_tensor_to_pil_img(inpainted)

    naive_merge, merged, inpainted = seg_and_inpaint(
        swap_tensor, outer_tensor, aot=True)

    # save inpainted
    # naive_merge = outer_tensor * inv_inner_gan_mask + \
    #     padded_swap_img * inner_gan_mask
    row = [
        naive_merge,
        merged,
        inpainted,
    ]
    create_img_row(row, 192).save(os.path.join(dir, 'inpaint.png'))

    # reinsert into full img
    print('reinsert')
    final_full, unaligned = solo_aligned_to_fam_final(
        inpainted,
        outer_quad,
        full_img,
        debug=True,
    )

    # save final
    unaligned.save(os.path.join(dir, 'unaligned.png'))
    final_full.save(os.path.join(dir, 'final.png'))


def ft_student_test(id, full_path, face_idx, gender):
    dir = os.path.join(FOLDER, 'ft-student', id)
    os.makedirs(dir, exist_ok=True)

    gender = torch.tensor(gender).to('cuda').unsqueeze(0)

    # crop and align
    print('crop and align')
    full_img, aligned, inner_quad = align(full_path, face_idx)
    aligned_256 = aligned.resize((256, 256), Image.LANCZOS)
    outer_aligned, outer_quad = outer_align(full_img, inner_quad)
    outer_aligned_192 = outer_aligned.resize((192, 192), Image.LANCZOS)

    # convert to tensor
    img_tensor = pil_to_tensor(aligned)
    img_tensor_256 = pil_to_tensor(aligned_256)
    outer_tensor = pil_to_tensor(outer_aligned)

    # invert
    print('e4e inversion')
    E = build_pretrained_e4e()
    G = build_pretrained_sg2().synthesis
    with torch.no_grad():
        e4e_w = E(img_tensor_256)

    # pti
    print('finetune generator')
    pti_trainer = PtiTrainer('cuda')
    pti_G = pti_trainer.train(img_tensor, e4e_w).synthesis
    with torch.no_grad():
        pti_inverted_tensor, pti_inverted = run_G(
            pti_G, e4e_w, return_tensor=True)

        # classify swap
        classify_lerper, classify_swaps = get_and_run_lerper(
            'lerp/5/5', MAGS, pti_G, e4e_w, gender)

    # finetune student
    student, samples, debug_data = finetune_student(
        pti_G, classify_lerper, e4e_w, gender)
    create_img_grid(samples, 256).save(os.path.join(dir, 'samples.png'))
    plt.plot(debug_data)
    plt.savefig(os.path.join(dir, 'loss.png'))
    plt.clf()

    # finetune student (adversarial)
    student, samples, rec_losses, gan_losses = finetune_student_adv(
        pti_G, classify_lerper, e4e_w, gender)
    create_img_grid(samples, 256).save(os.path.join(dir, 'samples_adv.png'))
    plt.plot(rec_losses)
    plt.savefig(os.path.join(dir, 'rec_loss.png'))
    plt.clf()
    plt.plot(gan_losses)
    plt.savefig(os.path.join(dir, 'gan_loss.png'))
    plt.clf()



def inpaint_test(id, full_path, face_idx, gender):
    dir = os.path.join(FOLDER, 'inpaint', id)
    os.makedirs(dir, exist_ok=True)

    gender = torch.tensor(gender).to('cuda').unsqueeze(0)

    # crop and align
    print('crop and align')
    full_img, aligned, inner_quad = align(full_path, face_idx)
    aligned_256 = aligned.resize((256, 256), Image.LANCZOS)
    outer_aligned, outer_quad = outer_align(full_img, inner_quad)
    outer_aligned_192 = outer_aligned.resize((192, 192), Image.LANCZOS)

    # convert to tensor
    img_tensor = pil_to_tensor(aligned)
    img_tensor_256 = pil_to_tensor(aligned_256)
    outer_tensor = pil_to_tensor(outer_aligned)

    # invert
    print('e4e inversion')
    E = build_pretrained_e4e()
    G = build_pretrained_sg2().synthesis
    with torch.no_grad():
        e4e_w = E(img_tensor_256)

    # pti
    print('finetune generator')
    pti_trainer = PtiTrainer('cuda')
    pti_G = pti_trainer.train(img_tensor, e4e_w).synthesis
    with torch.no_grad():
        pti_inverted_tensor, pti_inverted = run_G(
            pti_G, e4e_w, return_tensor=True)

        # classify swap
        classify_lerper, classify_swaps = get_and_run_lerper(
            'lerp/5/5', MAGS, pti_G, e4e_w, gender)

    # finetune classify lerper on pti model
    print('finetune lerper')
    ft_classify_lerper = finetune_lerper(
        pti_G, classify_lerper, pti_inverted_tensor, e4e_w, gender)

    with torch.no_grad():
        # finetuned classify swap
        print('finetuned classify swap')
        ft_pti_classify_swaps = run_lerper(
            ft_classify_lerper, MAGS, pti_G, e4e_w, gender)

        best_swap = ft_pti_classify_swaps[3]
        swap_tensor = pil_to_tensor(best_swap)

        # seg and inpaint
        naive_merge1, merged1, inpainted1 = seg_and_inpaint(
            swap_tensor, outer_tensor, aot=False)
        create_img_row([
            naive_merge1,
            merged1,
            inpainted1,
        ], 192).save(os.path.join(dir, 'inpaint1.png'))

        naive_merge2, merged2, inpainted2 = seg_and_inpaint(
            swap_tensor, outer_tensor, aot=True)
        create_img_row([
            naive_merge2,
            merged2,
            inpainted2,
        ], 512).save(os.path.join(dir, 'inpaint2.png'))


if __name__ == '__main__':
    # run('sera13', '/home/asiu/data/sera/og/13.jpg', 1, 1.)
    # inpaint_test('sera13', '/home/asiu/data/sera/og/13.jpg', 1, 1.)
    # ft_student_test('sera13', '/home/asiu/data/sera/og/13.jpg', 1, 1.)

    # lafleur_path = '/home/asiu/data/tmp/e2e/lafleur-small/og_full.jpg'
    # run('lafleur-small', lafleur_path, 0, 1.)
    # inpaint_test('lafleur-small', lafleur_path, 0, 1.)
    # ft_student_test('lafleur-small', lafleur_path, 0, 1.)

    # lafleur_path = '/home/asiu/data/tmp/e2e/lafleur-large/og_full.jpg'
    # run('lafleur-large', lafleur_path, 0, 1.)
    # inpaint_test('lafleur-large', lafleur_path, 0, 1.)
    # ft_student_test('lafleur-large', lafleur_path, 0, 1.)

    run('sera12', '/home/asiu/data/sera/og/12.jpg', 0, 1.)
