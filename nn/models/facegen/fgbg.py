#!/usr/bin/env python3
import torch
import torch.nn as nn
from external.sg2 import persistence
from external.sg2.unit import SynthesisNetwork, MappingNetwork, SynthesisBlock, \
    Conv2dLayer, SynthesisLayer, ToRGBLayer
import numpy as np
import external.sg2.misc as misc
from external.op import upfirdn2d


@persistence.persistent_class
class BgSynthesisNetwork(SynthesisNetwork):
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
        # return img
        return x


@persistence.persistent_class
class BlendBlock(nn.Module):
    def __init__(self, nc, k=1):
        super().__init__()

        self.to_mask = Conv2dLayer(
            nc,
            1,
            k,
            activation='sigmoid',
        )

    def forward(self, fg, bg):
        mask = self.to_mask(fg)
        return fg * (1. - mask) + bg * mask, mask


@persistence.persistent_class
class BlendSynthesisBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        w_dim,
        resolution,
        img_channels,
        architecture='skip',
        resample_filter=[1,3,3,1],
        conv_clamp=None,
        use_fp16=False,
        fp16_channels_last=False,
        **layer_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer(
            'resample_filter',
            upfirdn2d.setup_filter(resample_filter),
        )
        self.num_conv = 0
        self.num_torgb = 0

        self.conv0 = SynthesisLayer(
            in_channels,
            out_channels,
            w_dim=w_dim,
            resolution=resolution,
            up=2,
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

        self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
            conv_clamp=conv_clamp, channels_last=self.channels_last)
        self.num_torgb += 1

        self.blender = BlendBlock(out_channels)


    def forward(self,
        bg,
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

        misc.assert_shape(x, [
            None,
            self.in_channels,
            self.resolution // 2,
            self.resolution // 2,
        ])
        x = x.to(dtype=dtype, memory_format=memory_format)

        x = self.conv0(
            x,
            next(w_iter),
            fused_modconv=fused_modconv,
            **layer_kwargs,
        )
        x, seg = self.blender(x, bg)
        x = self.conv1(
            x,
            next(w_iter),
            fused_modconv=fused_modconv,
            **layer_kwargs,
        )


        if img is not None:
            misc.assert_shape(img, [
                None,
                self.img_channels,
                self.resolution // 2,
                self.resolution // 2,
            ])
            img = upfirdn2d.upsample2d(img, self.resample_filter)

        y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
        y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img, seg


@persistence.persistent_class
class NonUpSynthesisBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        w_dim,
        resolution,
        img_channels,
        is_last,
        architecture='skip',
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
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)

        self.num_conv = 0
        self.num_torgb = 0

        assert in_channels > 0

        self.conv0 = SynthesisLayer(
            in_channels,
            out_channels,
            w_dim=w_dim,
            resolution=resolution,
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

        if architecture == 'resnet':
            self.skip = Conv2dLayer(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
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
        misc.assert_shape(x, [
            None,
            self.in_channels,
            self.resolution,
            self.resolution,
        ])
        x = x.to(dtype=dtype, memory_format=memory_format)

        if self.architecture == 'resnet':
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
            misc.assert_shape(img, [
                None,
                self.img_channels,
                self.resolution,
                self.resolution,
            ])
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img


@persistence.persistent_class
class FgSynthesisNetwork(nn.Module):
    def __init__(self,
        w_dim,
        img_resolution,
        img_channels,
        channel_base=32768,
        channel_max=512,
        num_fp16_res=0,
        additional_resolutions=[],
        **block_kwargs,
    ):
        assert img_resolution >= 4 and \
            img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.additional_resolutions = additional_resolutions
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels

        # self.block_resolutions = [
        #     2 ** i for i in range(2, self.img_resolution_log2 + 1)
        # ]
        self.block_resolutions = []
        for i in range(2, self.img_resolution_log2 + 1):
            res = 2**i
            self.block_resolutions.append((res, f'b{res}'))
            if res in additional_resolutions:
                self.block_resolutions.append((res, f'b{res}_2'))

        # channels_dict = {
        #     res: min(channel_base // res, channel_max) \
        #     for res in self.block_resolutions
        # }
        channels_dict = {}
        for res, _ in self.block_resolutions:
            if res not in channels_dict:
                channels_dict[res] = min(channel_base // res, channel_max)

        fp16_resolution = max(
            2 ** (self.img_resolution_log2 + 1 - num_fp16_res),
            8,
        )

        self.num_ws = 0
        for res, label in self.block_resolutions:
            is_additional = label.endswith('_2')
            if is_additional:
                in_channels = channels_dict[res]
            else:
                in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            if is_last:
                assert not is_additional
                block = BlendSynthesisBlock(
                    in_channels,
                    out_channels,
                    w_dim=w_dim,
                    resolution=res,
                    img_channels=img_channels,
                    use_fp16=use_fp16,
                    **block_kwargs,
                )
            else:
                if is_additional:
                    block = NonUpSynthesisBlock(
                        in_channels,
                        out_channels,
                        w_dim=w_dim,
                        resolution=res,
                        img_channels=img_channels,
                        use_fp16=use_fp16,
                        **block_kwargs,
                    )
                else:
                    block = SynthesisBlock(
                        in_channels,
                        out_channels,
                        w_dim=w_dim,
                        resolution=res,
                        img_channels=img_channels,
                        is_last=False,
                        use_fp16=use_fp16,
                        **block_kwargs,
                    )
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, label, block)

    def forward(self, bg, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res, label in self.block_resolutions:
                block = getattr(self, label)
                block_ws.append(ws.narrow(
                    1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for i, (res_label, cur_ws) in enumerate(zip(
            self.block_resolutions,
            block_ws,
        )):
            block = getattr(self, res_label[1])
            if i == len(self.block_resolutions) - 1:
                x, img, seg = block(bg, x, img, cur_ws, **block_kwargs)
            else:
                x, img = block(x, img, cur_ws, **block_kwargs)

        return img, seg


@persistence.persistent_class
class FgBgGenerator(nn.Module):
    def __init__(self, cfg, additional_resolutions=[]):
        super().__init__()
        self.imsize = cfg.dataset.imsize
        self.z_dims_fg = 512
        self.z_dims_bg = 256

        self.g_fg = FgSynthesisNetwork(
            w_dim=self.z_dims_fg,
            img_resolution=self.imsize,
            img_channels=3,
            channel_base=64 * self.imsize,
            channel_max=self.z_dims_fg,
            num_fp16_res=4,
            conv_clamp=256,
            fp16_channels_last=False,
            architecture='resnet',
            additional_resolutions=additional_resolutions,
        )
        self.g_bg = BgSynthesisNetwork(
            w_dim=self.z_dims_bg,
            img_resolution=self.imsize,
            img_channels=3,
            channel_base=64 * self.imsize,
            channel_max=self.z_dims_bg,
            num_fp16_res=4,
            conv_clamp=256,
            fp16_channels_last=False,
            architecture='resnet',
        )

        self.f_fg = MappingNetwork(
            z_dim=self.z_dims_fg,
            c_dim=0,
            w_dim=self.z_dims_fg,
            num_ws=self.g_fg.num_ws,
            num_layers=8,
        )
        self.f_bg = MappingNetwork(
            z_dim=self.z_dims_bg,
            c_dim=0,
            w_dim=self.z_dims_bg,
            num_ws=self.g_bg.num_ws,
            num_layers=4,
        )

    def forward(self,
        z_fg,
        z_bg,
        truncation_psi=1,
        truncation_cutoff=None,
    ):
        ws_bg = self.f_bg(
            z_bg,
            None,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
        )
        x_bg = self.g_bg(ws_bg)

        ws_fg = self.f_fg(
            z_fg,
            None,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
        )
        img, seg = self.g_fg(x_bg, ws_fg)
        return img, seg

    def prep_for_train_phase(self):
        self.requires_grad_(True)
