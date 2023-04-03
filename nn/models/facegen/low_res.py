#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import external.sg2.misc as misc
from external.sg2 import persistence
from external.op import upfirdn2d
from external.sg2.unit import SynthesisLayer, ToRGBLayer, Conv2dLayer


@persistence.persistent_class
class OptionalUpSynthesisBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        w_dim,
        resolution,
        img_channels,
        is_last,
        is_up,
        architecture='skip',
        resample_filter=[1,3,3,1],
        conv_clamp=None,
        use_fp16=False,
        fp16_channels_last=False,
        **layer_kwargs,
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.is_up = is_up
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer(
            'resample_filter',
            upfirdn2d.setup_filter(resample_filter),
        )
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = nn.Parameter(
                torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=resolution,
                up=2 if is_up else 1,
                resample_filter=resample_filter,
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
                **layer_kwargs,
            )
            self.num_conv += 1

        self.conv1 = SynthesisLayer(
            out_channels,
            out_channels,
            w_dim=w_dim,
            resolution=resolution,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
            **layer_kwargs,
        )
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                up=2 if is_up else 1,
                resample_filter=resample_filter,
                channels_last=self.channels_last,
            )

    def forward(self,
        x,
        img,
        ws,
        force_fp32=False,
        fused_modconv=None,
        **layer_kwargs,
    ):
        misc.assert_shape(
            ws,
            [None, self.num_conv + self.num_torgb, self.w_dim],
        )
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else \
            torch.float32
        memory_format = torch.channels_last \
            if self.channels_last and not force_fp32 else \
            torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # treat as a constant
                fused_modconv = (not self.training) and \
                    (dtype == torch.float32 or int(x.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(
                x,
                next(w_iter),
                fused_modconv=fused_modconv,
                **layer_kwargs,
            )
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(
                x,
                next(w_iter),
                fused_modconv=fused_modconv,
                **layer_kwargs,
            )
            x = self.conv1(
                x,
                next(w_iter),
                fused_modconv=fused_modconv,
                gain=np.sqrt(0.5),
                **layer_kwargs,
            )
            x = y.add_(x)
        else:
            x = self.conv0(
                x,
                next(w_iter),
                fused_modconv=fused_modconv,
                **layer_kwargs,
            )
            x = self.conv1(
                x,
                next(w_iter),
                fused_modconv=fused_modconv,
                **layer_kwargs,
            )

        # ToRGB.
        if img is not None:
            if self.is_up:
                img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img


@persistence.persistent_class
class LowResSynthesisNetwork(nn.Module):
    def __init__(self,
        low_res,
        w_dim=512,
        og_res=1024,
        img_channels=3,
        channel_base=32768,
        channel_max=512,
        num_fp16_res=0,
        **block_kwargs,
    ):
        super().__init__()
        self.low_res = low_res
        self.w_dim = w_dim
        self.og_res = og_res
        self.img_resolution_log2 = int(np.log2(og_res))
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
            is_last = (res == self.og_res)
            block = OptionalUpSynthesisBlock(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=res if res <= low_res else low_res,
                img_channels=img_channels,
                is_last=is_last,
                is_up=res <= low_res,
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
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img


@persistence.persistent_class
class LowResGenerator(nn.Module):
    def __init__(self, cfg, nc_base=32):
        super().__init__()
        low_res = cfg.dataset.imsize

        self.net = LowResSynthesisNetwork(
            low_res,
            channel_base=nc_base * 1024,
            num_fp16_res=4,
            conv_clamp=256,
        )

    def forward(self, ws):
        out = self.net(ws, noise_mode='const')
        return out

    def prep_for_train_phase(self):
        self.net.requires_grad_(True)
