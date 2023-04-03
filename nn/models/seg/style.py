#!/usr/bin/env python3
import torch
import torch.nn as nn
from external.sg2 import persistence
from ai_old.util.params import init_params
from ai_old.util.pretrained import build_pretrained_sg2
from ai_old.util.etc import resize_imgs


@persistence.persistent_class
class _SegBlock(nn.Module):
    def __init__(self, nc1, is_first, is_last, nc2=16, n_labels=4):
        super().__init__()
        assert not (is_first and is_last)
        self.is_first = is_first

        self.conv1 = nn.Conv2d(nc1, nc2, kernel_size=1, stride=1)

        if not self.is_first:
            self.conv2 = nn.Sequential(
                nn.Conv2d(nc2, nc2, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
            )

        if is_last:
            self.final = nn.Conv2d(nc2, n_labels, kernel_size=1, stride=1)
        else:
            self.final = nn.Upsample(scale_factor=2)

    def forward(self, res, feats):
        feats = feats.to(
            dtype=torch.float32,
            memory_format=torch.contiguous_format,
        )
        x = self.conv1(feats)
        if not self.is_first:
            x = self.conv2(x + res)
        x = self.final(x)
        return x


@persistence.persistent_class
class Segmenter(nn.Module):
    def __init__(self, seg_imsize=128):
        super().__init__()
        self.seg_imsize = seg_imsize

        channels_per_block = [512, 512, 256, 128, 64, 32]
        n_blocks = len(channels_per_block)
        blocks = []
        for i, nc in enumerate(channels_per_block):
            blocks.append(_SegBlock(
                nc1=nc,
                is_first=i == 0,
                is_last=i == n_blocks - 1,
            ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, feats):
        x = None
        for i, block in enumerate(self.blocks):
            x = block(x, feats[i])
            i += 2
        return resize_imgs(x, self.seg_imsize)


@persistence.persistent_class
class SegmenterAndGenerator(nn.Module):
    def __init__(self,
        cfg,
        swapper_type='per_latent',
    ):
        super().__init__()

        G = build_pretrained_sg2(g_type='seg')
        self.g = G.synthesis
        self.g.requires_grad_(False)

        self.s = Segmenter()
        self.s.apply(init_params())

    def forward(self, w_plus):
        img, feats = self.g(w_plus)
        seg = self.s(feats)
        return img, seg

    def prep_for_train_phase(self):
        self.s.requires_grad_(True)
