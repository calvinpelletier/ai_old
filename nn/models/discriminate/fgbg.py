#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from external.sg2 import persistence
from ai_old.nn.models.discriminate.fast_sg2 import FastSg2Discriminator


@persistence.persistent_class
class FgBgDiscriminator(nn.Module):
    def __init__(self, cfg, combo_imsize=64):
        super().__init__()

        self.main = FastSg2Discriminator(
            cfg,
            nc_in=3,
            nc_base=64,
            nc_max=512,
            num_fp16_res=4,
        )

        self.combo_imsize = combo_imsize
        self.combo = FastSg2Discriminator(
            cfg,
            nc_in=4,
            nc_base=16,
            nc_max=256,
            num_fp16_res=0,
            imsize_override=self.combo_imsize,
        )

    def forward(self, img, seg):
        main_out = self.main(img)

        combo = F.interpolate(
            torch.cat([img, seg], dim=1),
            size=(self.combo_imsize, self.combo_imsize),
            mode='bilinear',
            align_corners=True,
        )
        combo_out = self.combo(combo)

        return main_out + combo_out


    def prep_for_train_phase(self):
        self.requires_grad_(True)
