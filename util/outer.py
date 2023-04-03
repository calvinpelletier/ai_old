#!/usr/bin/env python3
import torch
import cv2
from ai_old.util.etc import resize_imgs
from ai_old.util.face import custom_align_face
from ai_old.util.inverse import get_outer_quad


DI_K_192 = 16
OUTER_BOUNDARY_DIV = 32


def get_di_k(imsize):
    return int(DI_K_192 * (imsize / 192))


def get_dilate_kernel(imsize):
    di_k = get_di_k(imsize)
    return cv2.getStructuringElement(cv2.MORPH_RECT, (di_k, di_k))


def get_outer_boundary_size(imsize):
    return imsize // OUTER_BOUNDARY_DIV


def get_outer_boundary_mask(imsize):
    outer_boundary_mask = torch.zeros(imsize, imsize)
    obs = get_outer_boundary_size(imsize)
    for y in range(imsize):
        for x in range(imsize):
            if x < obs or x >= imsize - obs or \
                    y < obs or y >= imsize - obs:
                outer_boundary_mask[y][x] = 1.
    return outer_boundary_mask


def get_inner_mask(inner_imsize, outer_imsize, device='cuda'):
    inner_mask = torch.zeros(outer_imsize, outer_imsize, device=device)
    half_delta = (outer_imsize - inner_imsize) // 2
    for y in range(inner_imsize):
        for x in range(inner_imsize):
            inner_mask[y + half_delta][x + half_delta] = 1.
    return inner_mask


def dilate_mask(mask, dilate_kernel):
    dilated_mask = cv2.dilate(mask.cpu().numpy() * 255., dilate_kernel)
    dilated_mask = torch.tensor(dilated_mask / 255.).to('cuda')
    return (dilated_mask > 0.5).float()


def fhbc_seg_to_facehair(seg):
    return torch.bitwise_or(
        seg == 0, # face
        seg == 1, # hair
    ).float()


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


def outer_align(full_img, inner_quad):
    outer_quad = get_outer_quad(inner_quad, full=full_img)
    outer_imsize = 1024 + 512
    outer_aligned = custom_align_face(full_img, outer_quad, outer_imsize)
    return outer_aligned, outer_quad
