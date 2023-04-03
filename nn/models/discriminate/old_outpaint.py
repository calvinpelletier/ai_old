#!/usr/bin/env python3
import torch.nn as nn
from external.sg2.unit import DiscriminatorBlock, DiscriminatorEpilogue
import numpy as np
from ai_old.util.etc import resize_imgs, nearest_lower_power_of_2
from external.sg2 import persistence


@persistence.persistent_class
class OutpaintDiscriminator(nn.Module):
    def __init__(self,
        cfg,
        nc_in=3,
        nc_base=32,
        nc_max=512,
        num_fp16_res=4,
        imsize_override=None,
    ):
        super().__init__()
        if imsize_override is None:
            imsize = cfg.dataset.imsize
        else:
            imsize = imsize_override
        imsize = nearest_lower_power_of_2(imsize)
        self.imsize = imsize

        log2_imsize = int(np.log2(imsize))
        self.block_resolutions = [
            2 ** i for i in range(log2_imsize, 2, -1)
        ]
        channels_dict = {
            res: min((nc_base * imsize) // res, nc_max) \
            for res in self.block_resolutions + [4]
        }
        fp16_resolution = max(
            2 ** (log2_imsize + 1 - num_fp16_res),
            8,
        )
        mbstd = min(cfg.dataset.batch_size // cfg.num_gpus, 4)

        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < imsize else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(
                in_channels,
                tmp_channels,
                out_channels,
                resolution=res,
                first_layer_idx=cur_layer_idx,
                use_fp16=use_fp16,
                img_channels=nc_in,
                architecture='resnet',
                conv_clamp=256,
                freeze_layers=False,
                fp16_channels_last=False,
            )
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

        self.b4 = DiscriminatorEpilogue(
            channels_dict[4],
            cmap_dim=0,
            resolution=4,
            img_channels=nc_in,
            architecture='resnet',
            conv_clamp=256,
            mbstd_group_size=mbstd,
        )

    def forward(self, img, **block_kwargs):
        img = resize_imgs(img, self.imsize)

        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        x = self.b4(x, img, None)
        return x

    def prep_for_train_phase(self):
        self.requires_grad_(True)
