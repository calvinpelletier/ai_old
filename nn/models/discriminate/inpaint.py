#!/usr/bin/env python3
import torch
from external.sg2 import persistence
from ai_old.nn.models.discriminate.fast_sg2 import FastSg2Discriminator


@persistence.persistent_class
class InpaintDiscriminator(FastSg2Discriminator):
    def __init__(self,
        cfg,
        include_mask=True,
        nc_base=32,
        nc_max=512,
        num_fp16_res=4,
    ):
        super().__init__(
            cfg,
            nc_in=4 if include_mask else 3,
            nc_base=nc_base,
            nc_max=nc_max,
            num_fp16_res=num_fp16_res,
        )

        self.include_mask = include_mask

    def forward(self, img, mask, **block_kwargs):
        if self.include_mask:
            img = torch.cat([img, mask.unsqueeze(1)], dim=1)

        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        x = self.b4(x, img, None)
        return x
