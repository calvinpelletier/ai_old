#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from external.sg2 import persistence
from ai_old.util.params import init_params


@persistence.persistent_class
class AotInpainter(nn.Module):
    def __init__(self, cfg, rates='1+2+4+8', n_blocks=8):
        super().__init__()
        assert cfg.dataset.imsize == 512
        rates = list(map(int, list(rates.split('+'))))

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, 64, 7),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True)
        )

        self.middle = nn.Sequential(*[
            AOTBlock(256, rates) for _ in range(n_blocks)
        ])

        self.decoder = nn.Sequential(
            UpConv(256, 128),
            nn.ReLU(True),
            UpConv(128, 64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )

        self.apply(init_params())

    def forward(self, full, mask):
        mask = mask.unsqueeze(1)
        inv_mask = 1. - mask
        input = (full * inv_mask) + mask

        x = torch.cat([input, mask], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.tanh(x)

        out = full * inv_mask + x * mask
        return out

    def prep_for_train_phase(self):
        self.requires_grad_(True)


class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(
            x,
            scale_factor=2,
            mode='bilinear',
            align_corners=True,
        ))


class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super().__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim//4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [
            self.__getattr__(f'block{str(i).zfill(2)}')(x) \
            for i in range(len(self.rates))
        ]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = aot_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def aot_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat
