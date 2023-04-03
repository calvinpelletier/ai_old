#!/usr/bin/env python3
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Augmenter(nn.Module):
    def __init__(self, prob, types):
        super().__init__()
        self.prob = prob
        self.types = types

    def forward(self, imgs, hflip=False):
        if random.random() < self.prob:
            if hflip and random.random() < 0.5:
                imgs = torch.flip(imgs, dims=(3,))
            imgs = _diff_augment(imgs, aug_types=self.types)
        return imgs

def _diff_augment(imgs, aug_types=[]):
    for p in aug_types:
        for f in AUGMENT_FNS[p]:
            imgs = f(imgs)
    return imgs.contiguous()

def _rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x

def _rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x

def _rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x

def _rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x

def _rand_offset(x, ratio=1, ratio_h=1, ratio_v=1):
    w, h = x.size(2), x.size(3)

    for image_index in range(x.size(0)):
        max_h = int(w * ratio * ratio_h)
        max_v = int(h * ratio * ratio_v)

        value_h = random.randint(0, max_h) * 2 - max_h
        value_v = random.randint(0, max_v) * 2 - max_v

        if abs(value_h) > 0:
            x[image_index] = torch.roll(x[image_index], value_h, 2)

        if abs(value_v) > 0:
            x[image_index] = torch.roll(x[image_index], value_v, 1)

    return x

def _rand_offset_h(x, ratio=1):
    return _rand_offset(x, ratio=1, ratio_h=ratio, ratio_v=0)

def _rand_offset_v(x, ratio=1):
    return _rand_offset(x, ratio=1, ratio_h=0, ratio_v=ratio)

def _rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x

AUGMENT_FNS = {
    'color': [_rand_brightness, _rand_saturation, _rand_contrast],
    'offset': [_rand_offset],
    'offset_h': [_rand_offset_h],
    'offset_v': [_rand_offset_v],
    'translation': [_rand_translation],
    'cutout': [_rand_cutout],
}
