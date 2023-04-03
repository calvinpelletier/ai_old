#!/usr/bin/env python3
import torch
import torch.nn as nn
from external.sg2 import persistence
import external.sg2.misc as misc
from external.sg2.unit import modulated_conv2d, ToRGBLayer, Conv2dLayer, \
    SynthesisLayer, MappingNetwork
from external.op import bias_act
from external.op import upfirdn2d
import numpy as np

# for generating three images with the same input noise (e.g. for brew training)


@persistence.persistent_class
class TriSynthesisLayer(SynthesisLayer):
    def forward(self,
        x1, x2, x3,
        w1, w2, w3,
        noise_mode='random',
        fused_modconv=True,
        gain=1,
    ):
        assert noise_mode in ['random', 'const', 'none']
        styles1 = self.affine(w1)
        styles2 = self.affine(w2)
        styles3 = self.affine(w3)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn(
                [x1.shape[0], 1, self.resolution, self.resolution],
                device=x1.device,
            ) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        x1 = modulated_conv2d(
            x=x1,
            weight=self.weight,
            styles=styles1,
            noise=noise,
            up=self.up,
            padding=self.padding,
            resample_filter=self.resample_filter,
            flip_weight=flip_weight,
            fused_modconv=fused_modconv,
        )
        x2 = modulated_conv2d(
            x=x2,
            weight=self.weight,
            styles=styles2,
            noise=noise,
            up=self.up,
            padding=self.padding,
            resample_filter=self.resample_filter,
            flip_weight=flip_weight,
            fused_modconv=fused_modconv,
        )
        x3 = modulated_conv2d(
            x=x3,
            weight=self.weight,
            styles=styles3,
            noise=noise,
            up=self.up,
            padding=self.padding,
            resample_filter=self.resample_filter,
            flip_weight=flip_weight,
            fused_modconv=fused_modconv,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain \
            if self.conv_clamp is not None else None
        x1 = bias_act.bias_act(
            x1,
            self.bias.to(x1.dtype),
            act=self.activation,
            gain=act_gain,
            clamp=act_clamp,
        )
        x2 = bias_act.bias_act(
            x2,
            self.bias.to(x2.dtype),
            act=self.activation,
            gain=act_gain,
            clamp=act_clamp,
        )
        x3 = bias_act.bias_act(
            x3,
            self.bias.to(x3.dtype),
            act=self.activation,
            gain=act_gain,
            clamp=act_clamp,
        )
        return x1, x2, x3


@persistence.persistent_class
class TriSynthesisBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        w_dim,
        resolution,
        img_channels,
        is_last,
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
            self.conv0 = TriSynthesisLayer(
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

        self.conv1 = TriSynthesisLayer(
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
                up=2,
                resample_filter=resample_filter,
                channels_last=self.channels_last,
            )

    def forward(self,
        x1,
        x2,
        x3,
        img1,
        img2,
        img3,
        ws1,
        ws2,
        ws3,
        force_fp32=False,
        fused_modconv=None,
        **layer_kwargs,
    ):
        w_iter1 = iter(ws1.unbind(dim=1))
        w_iter2 = iter(ws2.unbind(dim=1))
        w_iter3 = iter(ws3.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else \
            torch.float32
        memory_format = torch.channels_last \
            if self.channels_last and not force_fp32 else \
            torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # treat as a constant
                fused_modconv = (not self.training) and \
                    (dtype == torch.float32 or int(x1.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws1.shape[0], 1, 1, 1])
            x1 = x
            x2 = x
            x3 = x
        else:
            x1 = x1.to(dtype=dtype, memory_format=memory_format)
            x2 = x2.to(dtype=dtype, memory_format=memory_format)
            x3 = x3.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x1, x2, x3 = self.conv1(
                x1, x2, x3,
                next(w_iter1), next(w_iter2), next(w_iter3),
                fused_modconv=fused_modconv,
                **layer_kwargs,
            )

        elif self.architecture == 'resnet':
            y1 = self.skip(x1, gain=np.sqrt(0.5))
            y2 = self.skip(x2, gain=np.sqrt(0.5))
            y3 = self.skip(x3, gain=np.sqrt(0.5))
            x1, x2, x3 = self.conv0(
                x1, x2, x3,
                next(w_iter1), next(w_iter2), next(w_iter3),
                fused_modconv=fused_modconv,
                **layer_kwargs,
            )
            x1, x2, x3 = self.conv1(
                x1, x2, x3,
                next(w_iter1), next(w_iter2), next(w_iter3),
                fused_modconv=fused_modconv,
                gain=np.sqrt(0.5),
                **layer_kwargs,
            )
            x1 = y2.add_(x1)
            x2 = y2.add_(x2)
            x3 = y3.add_(x3)

        else:
            x1, x2, x3 = self.conv0(
                x1, x2, x3,
                next(w_iter1), next(w_iter2), next(w_iter3),
                fused_modconv=fused_modconv,
                **layer_kwargs,
            )
            x1, x2, x3 = self.conv1(
                x1, x2, x3,
                next(w_iter1), next(w_iter2), next(w_iter3),
                fused_modconv=fused_modconv,
                **layer_kwargs,
            )

        # ToRGB.
        if img1 is not None:
            img1 = upfirdn2d.upsample2d(img1, self.resample_filter)
            img2 = upfirdn2d.upsample2d(img2, self.resample_filter)
            img3 = upfirdn2d.upsample2d(img3, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y1 = self.torgb(x1, next(w_iter1), fused_modconv=fused_modconv)
            y1 = y1.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img1 = img1.add_(y1) if img1 is not None else y1

            y2 = self.torgb(x2, next(w_iter2), fused_modconv=fused_modconv)
            y2 = y2.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img2 = img2.add_(y2) if img2 is not None else y2

            y3 = self.torgb(x3, next(w_iter3), fused_modconv=fused_modconv)
            y3 = y3.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img3 = img3.add_(y3) if img3 is not None else y3

        return x1, x2, x3, img1, img2, img3


@persistence.persistent_class
class TriSynthesisNetwork(nn.Module):
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
            block = TriSynthesisBlock(
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

    def forward(self, ws1, ws2, ws3, **block_kwargs):
        block_ws1 = []
        block_ws2 = []
        block_ws3 = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws1, [None, self.num_ws, self.w_dim])
            misc.assert_shape(ws2, [None, self.num_ws, self.w_dim])
            misc.assert_shape(ws3, [None, self.num_ws, self.w_dim])
            ws1 = ws1.to(torch.float32)
            ws2 = ws2.to(torch.float32)
            ws3 = ws3.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws1.append(ws1.narrow(
                    1, w_idx, block.num_conv + block.num_torgb))
                block_ws2.append(ws2.narrow(
                    1, w_idx, block.num_conv + block.num_torgb))
                block_ws3.append(ws3.narrow(
                    1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x1 = x2 = x3 = img1 = img2 = img3 = None
        for res, cur_ws1, cur_ws2, cur_ws3 in zip(
            self.block_resolutions,
            block_ws1,
            block_ws2,
            block_ws3,
        ):
            block = getattr(self, f'b{res}')
            x1, x2, x3, img1, img2, img3 = block(
                x1, x2, x3,
                img1, img2, img3,
                cur_ws1, cur_ws2, cur_ws3,
                **block_kwargs,
            )
        return img1, img2, img3


@persistence.persistent_class
class TriGenerator(nn.Module):
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
        self.synthesis = TriSynthesisNetwork(
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
        z1,
        z2,
        z3,
        c1,
        c2,
        c3,
        truncation_psi=1,
        truncation_cutoff=None,
        **synthesis_kwargs,
    ):
        ws1 = self.mapping(
            z1,
            c1,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
        )
        ws2 = self.mapping(
            z1,
            c1,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
        )
        ws3 = self.mapping(
            z1,
            c1,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
        )
        img1, img2, img3 = self.synthesis(ws1, ws2, ws3, **synthesis_kwargs)
        return img1, img2, img3


@persistence.persistent_class
class TriOptionalUpSynthesisBlock(nn.Module):
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
            self.conv0 = TriSynthesisLayer(
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

        self.conv1 = TriSynthesisLayer(
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
        x1,
        x2,
        x3,
        img1,
        img2,
        img3,
        ws1,
        ws2,
        ws3,
        force_fp32=False,
        fused_modconv=None,
        **layer_kwargs,
    ):
        w_iter1 = iter(ws1.unbind(dim=1))
        w_iter2 = iter(ws2.unbind(dim=1))
        w_iter3 = iter(ws3.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else \
            torch.float32
        memory_format = torch.channels_last \
            if self.channels_last and not force_fp32 else \
            torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # treat as a constant
                fused_modconv = (not self.training) and \
                    (dtype == torch.float32 or int(x1.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws1.shape[0], 1, 1, 1])
            x1 = x
            x2 = x
            x3 = x
        else:
            x1 = x1.to(dtype=dtype, memory_format=memory_format)
            x2 = x2.to(dtype=dtype, memory_format=memory_format)
            x3 = x3.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x1, x2, x3 = self.conv1(
                x1, x2, x3,
                next(w_iter1), next(w_iter2), next(w_iter3),
                fused_modconv=fused_modconv,
                **layer_kwargs,
            )

        elif self.architecture == 'resnet':
            y1 = self.skip(x1, gain=np.sqrt(0.5))
            y2 = self.skip(x2, gain=np.sqrt(0.5))
            y3 = self.skip(x3, gain=np.sqrt(0.5))
            x1, x2, x3 = self.conv0(
                x1, x2, x3,
                next(w_iter1), next(w_iter2), next(w_iter3),
                fused_modconv=fused_modconv,
                **layer_kwargs,
            )
            x1, x2, x3 = self.conv1(
                x1, x2, x3,
                next(w_iter1), next(w_iter2), next(w_iter3),
                fused_modconv=fused_modconv,
                gain=np.sqrt(0.5),
                **layer_kwargs,
            )
            x1 = y2.add_(x1)
            x2 = y2.add_(x2)
            x3 = y3.add_(x3)

        else:
            x1, x2, x3 = self.conv0(
                x1, x2, x3,
                next(w_iter1), next(w_iter2), next(w_iter3),
                fused_modconv=fused_modconv,
                **layer_kwargs,
            )
            x1, x2, x3 = self.conv1(
                x1, x2, x3,
                next(w_iter1), next(w_iter2), next(w_iter3),
                fused_modconv=fused_modconv,
                **layer_kwargs,
            )

        # ToRGB.
        if img1 is not None:
            if self.is_up:
                img1 = upfirdn2d.upsample2d(img1, self.resample_filter)
                img2 = upfirdn2d.upsample2d(img2, self.resample_filter)
                img3 = upfirdn2d.upsample2d(img3, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y1 = self.torgb(x1, next(w_iter1), fused_modconv=fused_modconv)
            y1 = y1.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img1 = img1.add_(y1) if img1 is not None else y1

            y2 = self.torgb(x2, next(w_iter2), fused_modconv=fused_modconv)
            y2 = y2.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img2 = img2.add_(y2) if img2 is not None else y2

            y3 = self.torgb(x3, next(w_iter3), fused_modconv=fused_modconv)
            y3 = y3.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img3 = img3.add_(y3) if img3 is not None else y3

        return x1, x2, x3, img1, img2, img3


@persistence.persistent_class
class TriLowResSynthesisNetwork(nn.Module):
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
            block = TriOptionalUpSynthesisBlock(
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

    def forward(self, ws1, ws2, ws3, **block_kwargs):
        block_ws1 = []
        block_ws2 = []
        block_ws3 = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws1, [None, self.num_ws, self.w_dim])
            misc.assert_shape(ws2, [None, self.num_ws, self.w_dim])
            misc.assert_shape(ws3, [None, self.num_ws, self.w_dim])
            ws1 = ws1.to(torch.float32)
            ws2 = ws2.to(torch.float32)
            ws3 = ws3.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws1.append(ws1.narrow(
                    1, w_idx, block.num_conv + block.num_torgb))
                block_ws2.append(ws2.narrow(
                    1, w_idx, block.num_conv + block.num_torgb))
                block_ws3.append(ws3.narrow(
                    1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x1 = x2 = x3 = img1 = img2 = img3 = None
        for res, cur_ws1, cur_ws2, cur_ws3 in zip(
            self.block_resolutions,
            block_ws1,
            block_ws2,
            block_ws3,
        ):
            block = getattr(self, f'b{res}')
            x1, x2, x3, img1, img2, img3 = block(
                x1, x2, x3,
                img1, img2, img3,
                cur_ws1, cur_ws2, cur_ws3,
                **block_kwargs,
            )
        return img1, img2, img3
