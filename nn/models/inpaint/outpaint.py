#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_old.util.params import init_params
from ai_old.nn.blocks.res import FancyMultiLayerDownBlock, ResUpConvBlock
from ai_old.util.etc import log2_diff
from ai_old.nn.blocks.conv import ConvBlock
from ai_old.util.etc import resize_imgs, nearest_lower_power_of_2
from external.sg2 import persistence


def _get_masks(full_imsize, inner_imsize):
    inner_mask = torch.zeros(full_imsize, full_imsize)
    half_delta = (full_imsize - inner_imsize) // 2
    for y in range(inner_imsize):
        for x in range(inner_imsize):
            inner_mask[y + half_delta][x + half_delta] = 1.
    outer_mask = 1. - inner_mask
    return inner_mask, outer_mask


class _IIT(nn.Module):
    def __init__(self,
        imsize=256,
        smallest_imsize=8,
        nc_base=64,
        nc_max=512,
        nc_in=3,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()
        self.imsize = imsize

        n_down_up = log2_diff(imsize, smallest_imsize)
        nc = [min(nc_max, nc_base * (2 ** i)) for i in range(n_down_up + 1)]

        # initial conv (pre resize to model imsize)
        # self.initial = ConvBlock(
        #     nc_in,
        #     nc[0],
        #     norm='none',
        #     weight_norm=False,
        #     actv=actv,
        # )

        # main
        enc_blocks = [ConvBlock(
            nc_in,
            nc[0],
            norm='none',
            weight_norm=False,
            actv=actv,
        )]
        dec_blocks = []
        for i in range(n_down_up):
            enc_blocks.append(FancyMultiLayerDownBlock(
                nc[i],
                nc[i+1],
                n_layers=2,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            ))
            dec_blocks.append(ResUpConvBlock(
                nc[i+1],
                nc[i],
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            ))
        self.e = nn.Sequential(*enc_blocks)
        self.g = nn.Sequential(*dec_blocks[::-1])

        # final convs (post resize to input img size)
        self.final = nn.Sequential(
            ConvBlock(
                nc[0],
                nc[0],
                norm=norm,
                weight_norm=False,
                actv=actv,
            ),
            nn.Conv2d(nc[0], nc_in, kernel_size=1),
        )

    def forward(self, x):
        _, _, h, w = x.shape
        assert h == w
        # x = self.initial(x)
        x = resize_imgs(x, self.imsize)
        enc = self.e(x)
        out = self.g(enc)
        out = resize_imgs(out, h)
        out = self.final(out)
        return out


@persistence.persistent_class
class Outpainter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        full_imsize = cfg.dataset.imsize
        inner_imsize = cfg.dataset.inner_imsize

        model_imsize = nearest_lower_power_of_2(full_imsize)
        self.net = _IIT(imsize=model_imsize)
        self.net.apply(init_params())

        inner_mask, outer_mask = _get_masks(full_imsize, inner_imsize)
        self.register_buffer('inner_mask', inner_mask)
        self.register_buffer('outer_mask', outer_mask)

    def forward(self, full):
        padded_inner = full * self.inner_mask
        out = self.net(padded_inner)
        out = padded_inner + out * self.outer_mask
        return out

    def prep_for_train_phase(self):
        self.net.requires_grad_(True)


@persistence.persistent_class
class SmartOutpainter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        full_imsize = cfg.dataset.imsize
        inner_imsize = cfg.dataset.inner_imsize

        model_imsize = nearest_lower_power_of_2(full_imsize)
        self.net = _IIT(imsize=model_imsize)
        self.net.apply(init_params())

    def forward(self, full, mask):
        mask = mask.unsqueeze(1)
        inv_mask = 1. - mask
        input = full * inv_mask
        out = self.net(input)
        out = input + out * mask
        return out

    def prep_for_train_phase(self):
        self.net.requires_grad_(True)
