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


def get_masks(full_imsize, inner_imsize):
    inner_mask = torch.zeros(full_imsize, full_imsize)
    half_delta = (full_imsize - inner_imsize) // 2
    for y in range(inner_imsize):
        for x in range(inner_imsize):
            inner_mask[y + half_delta][x + half_delta] = 1.
    outer_mask = 1. - inner_mask
    return inner_mask, outer_mask


class _SegToSeg(nn.Module):
    def __init__(self,
        imsize=128,
        smallest_imsize=8,
        nc_in=7,
        nc_out=4,
        nc_base=16,
        nc_max=512,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()
        self.imsize = imsize

        n_down_up = log2_diff(imsize, smallest_imsize)
        nc = [min(nc_max, nc_base * (2 ** i)) for i in range(n_down_up + 1)]

        # outer blocks
        enc_blocks = [ConvBlock(
            nc_in,
            nc[0],
            norm='none',
            weight_norm=False,
            actv=actv,
        )]
        dec_blocks = [nn.Conv2d(nc[0], nc_out, kernel_size=1)]

        # inner blocks
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

    def forward(self, x):
        _, _, h, w = x.shape
        assert h == w
        x = resize_imgs(x, self.imsize)
        enc = self.e(x)
        out = self.g(enc)
        return resize_imgs(out, h)


@persistence.persistent_class
class OuterSegPredictor(nn.Module):
    def __init__(self,
        cfg,
        pred_from_seg_only=False,
        nc_base=16,
        nc_max=512,
    ):
        super().__init__()
        full_imsize = cfg.dataset.imsize
        inner_imsize = cfg.dataset.inner_imsize
        self.n_labels = cfg.dataset.n_labels
        self.pred_from_seg_only = pred_from_seg_only

        if self.pred_from_seg_only:
            nc_in = self.n_labels
        else:
            # concat img and seg
            nc_in = self.n_labels + 3

        model_imsize = nearest_lower_power_of_2(full_imsize)
        self.net = _SegToSeg(
            imsize=model_imsize,
            nc_in=nc_in,
            nc_out=self.n_labels,
            nc_base=nc_base,
            nc_max=nc_max,
        )
        self.net.apply(init_params())

        inner_mask, outer_mask = get_masks(full_imsize, inner_imsize)
        self.register_buffer('inner_mask', inner_mask)
        self.register_buffer('outer_mask', outer_mask)

    def forward(self, full_seg, full_img):
        full_seg = F.one_hot(full_seg, num_classes=self.n_labels)
        full_seg = full_seg.permute(0, 3, 1, 2)

        if self.pred_from_seg_only:
            padded_inner = full_seg * self.inner_mask
            out = self.net(padded_inner)
            out = padded_inner + out * self.outer_mask
        else:
            full_input = torch.cat([full_seg, full_img], dim=1)
            padded_inner = full_input * self.inner_mask
            out = self.net(padded_inner)
            out = full_seg * self.inner_mask + out * self.outer_mask
        return out

    def prep_for_train_phase(self):
        self.net.requires_grad_(True)


@persistence.persistent_class
class GatedOuterSegPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        full_imsize = cfg.dataset.imsize
        inner_imsize = cfg.dataset.inner_imsize
        self.n_labels = cfg.dataset.n_labels

        self.net = GatedGenerator(
            in_channels=n_labels,
            out_channels=n_labels,
            pretrained=False,
            final_tanh=False,
        )

        inner_mask, outer_mask = get_masks(full_imsize, inner_imsize)
        self.register_buffer('inner_mask', inner_mask)
        self.register_buffer('outer_mask', outer_mask)

    def forward(self, full_seg):
        full_seg = F.one_hot(full_seg, num_classes=self.n_labels)
        full_seg = full_seg.permute(0, 3, 1, 2)
        padded_inner_seg = full_seg * self.inner_mask
        out = self.net(full_seg, self.outer_mask)
        out = padded_inner_seg + out * self.outer_mask
        return out

    def prep_for_train_phase(self):
        self.net.requires_grad_(True)
