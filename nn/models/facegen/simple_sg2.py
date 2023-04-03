#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
from external.op import conv2d_resample_ref
from external.op import upfirdn2d
from external.op import bias_act
from external.op import fma
import external.sg2.misc as misc


def convert_fast_to_simple(fast_G):
    assert fast_G.w_dim == 512
    assert fast_G.img_resolution == 1024
    assert fast_G.img_channels == 3
    simple_G = SimpleSg2Generator(
        512,
        1024,
        3,
        channel_base=32768,
        channel_max=512,
        num_fp16_res=4,
        conv_clamp=256,
    ).to('cuda')
    simple_G.load_state_dict(fast_G.state_dict())
    simple_G.eval()
    return simple_G


# functionally equivalent to fast sg2 generator but able to be converted to onnx
class SimpleSg2Generator(nn.Module):
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

    def forward(self, ws):
        block_ws = []
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
            x, img = block(x, img, cur_ws, noise_mode='const')
        return img


class SynthesisBlock(torch.nn.Module):
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
            self.const = torch.nn.Parameter(
                torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
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
            misc.assert_shape(x, [
                None,
                self.in_channels,
                self.resolution // 2,
                self.resolution // 2,
            ])
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
            misc.assert_shape(img, [
                None,
                self.img_channels,
                self.resolution // 2,
                self.resolution // 2,
            ])
            img = upfirdn2d.upsample2d(img, self.resample_filter, impl='ref')
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img


class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        w_dim,
        resolution,
        kernel_size=3,
        up=1,
        use_noise=True,
        activation='lrelu',
        resample_filter=[1,3,3,1],
        conv_clamp=None,
        channels_last=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer(
            'resample_filter',
            upfirdn2d.setup_filter(resample_filter),
        )
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else \
            torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
        ]).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer(
                'noise_const',
                torch.randn([resolution, resolution]),
            )
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(
            x,
            [None, self.weight.shape[1], in_resolution, in_resolution],
        )
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn(
                [x.shape[0], 1, self.resolution, self.resolution],
                device=x.device,
            ) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
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
        x = bias_act.bias_act(
            x,
            self.bias.to(x.dtype),
            act=self.activation,
            gain=act_gain,
            clamp=act_clamp,
            impl='ref',
        )
        return x


class ToRGBLayer(torch.nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        w_dim,
        kernel_size=1,
        conv_clamp=None,
        channels_last=False,
    ):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else \
            torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
        ]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
            demodulate=False,
            fused_modconv=fused_modconv,
        )
        x = bias_act.bias_act(
            x,
            self.bias.to(x.dtype),
            clamp=self.conv_clamp,
            impl='ref',
        )
        return x


class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,
        out_features,
        bias=True,
        activation='linear',
        lr_multiplier=1,
        bias_init=0,
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(
            torch.randn([out_features, in_features]) / lr_multiplier,
        )
        self.bias = torch.nn.Parameter(
            torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation, impl='ref')
        return x


@misc.profiled_function
def modulated_conv2d(
    x,
    weight,
    styles,
    noise=None,
    up=1,
    down=1,
    padding=0,
    resample_filter=None,
    demodulate=True,
    flip_weight=True,
    fused_modconv=True,
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        # weight = weight * (1 / np.sqrt(in_channels * kh * kw) / \
        #     weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        # styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / \
            weight.norm(p=2, dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(p=2, dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        # dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
        dcoefs = ((w * w).sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample_ref.conv2d_resample(
            x=x,
            w=weight.to(x.dtype),
            f=resample_filter,
            up=up,
            down=down,
            padding=padding,
            flip_weight=flip_weight,
        )
        if demodulate and noise is not None:
            x = fma.fma(
                x,
                dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1),
                noise.to(x.dtype),
            )
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # treat as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample_ref.conv2d_resample(
        x=x,
        w=w.to(x.dtype),
        f=resample_filter,
        up=up,
        down=down,
        padding=padding,
        groups=batch_size,
        flip_weight=flip_weight,
    )
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x
