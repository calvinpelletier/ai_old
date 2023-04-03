#!/usr/bin/env python3
import torch.nn as nn
from external.sg2 import persistence
from external.sg2.unit import SynthesisNetwork, MappingNetwork


@persistence.persistent_class
class BlendGenerator(nn.Module):
    def __init__(self,
        cfg,
        imsize=128,
        z_dims=512,
        nc_in=3,
        nc_base=32,
        nc_max=512,
        num_fp16_res=4,
        f_n_layers=2,
    ):
        super().__init__()
        self.z_dims = z_dims
        self.g = BlendSynthesisNetwork(
            w_dim=z_dims,
            img_resolution=imsize,
            img_channels=nc_in,
            channel_base=nc_base * imsize,
            channel_max=nc_max,
            num_fp16_res=num_fp16_res,
            conv_clamp=256,
            fp16_channels_last=False,
        )
        self.num_ws = self.g.num_ws
        self.f = MappingNetwork(
            z_dim=z_dims,
            c_dim=0,
            w_dim=z_dims,
            num_ws=self.num_ws,
            num_layers=f_n_layers,
        )
        # self.b = BlendNetwork(
        #
        # )

    def forward(self,
        ibg,
        z,
        truncation_psi=1,
        truncation_cutoff=None,
        **synthesis_kwargs,
    ):
        ws = self.f(
            z,
            None,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
        )
        img = self.g(ibg, ws, **synthesis_kwargs)
        return img

    def prep_for_train_phase(self):
        self.requires_grad_(True)


@persistence.persistent_class
class BlendSynthesisNetwork(torch.nn.Module):
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
            block = BlendSynthesisBlock(
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

    def forward(self, ibg, ws, **block_kwargs):
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
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
            print(res, x.shape)
        print('~~~')
        return img
