#!/usr/bin/env python3
import torch
from external.sg2 import persistence
from external.sg2.unit import MappingNetwork, SynthesisBlock
import numpy as np
import external.sg2.misc as misc


@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,
        img_resolution,
        img_channels,
        channel_base=32768,
        channel_max=512,
        num_fp16_res=0,
        **block_kwargs,
    ):
        assert img_resolution >= 4 and \
            img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [
            2 ** i for i in range(2, self.img_resolution_log2 + 1)
        ]
        channels_dict = {
            res: min(channel_base // res, channel_max) \
            for res in self.block_resolutions
        }
        fp16_resolution = max(
            2 ** (self.img_resolution_log2 + 1 - num_fp16_res),
            8,
        )

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=res,
                img_channels=img_channels,
                is_last=is_last,
                use_fp16=use_fp16,
                **block_kwargs,
            )
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(
                    1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        feats = []
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
            if res >= 32:
                feats.append(x)
        return img, feats


@persistence.persistent_class
class GeneratorForSeg(torch.nn.Module):
    def __init__(self,
        z_dim,
        c_dim,
        w_dim,
        img_resolution,
        img_channels,
        mapping_kwargs={},
        synthesis_kwargs={},
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            **synthesis_kwargs,
        )
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(
            z_dim=z_dim,
            c_dim=c_dim,
            w_dim=w_dim,
            num_ws=self.num_ws,
            **mapping_kwargs,
        )

    def forward(self,
        z,
        c,
        truncation_psi=1,
        truncation_cutoff=None,
        **synthesis_kwargs,
    ):
        ws = self.mapping(
            z,
            c,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
        )
        img, feats = self.synthesis(ws, **synthesis_kwargs)
        return img, feats
