#!/usr/bin/env python3
import torch
import torch.nn as nn
from external.sg2 import persistence
from external.sg2.unit import FullyConnectedLayer, Conv2dLayer, modulated_conv2d
from ai_old.util.etc import log2_diff
from ai_old.nn.blocks.style_enc import StyleEncLayer, StyleResDownBlock
from external.op import upfirdn2d
import external.sg2.misc as misc
from external.op import bias_act
import numpy as np


@persistence.persistent_class
class FeatMapToLatent(nn.Module):
    def __init__(self,
        nc1,
        z_dims=512,
        down=2,
    ):
        super().__init__()
        nc2 = min(z_dims, nc1 * 2)

        self.conv = Conv2dLayer(
            nc1,
            nc2,
            kernel_size=3,
            down=down,
        )

        self.linear = FullyConnectedLayer(
            nc2,
            z_dims,
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.mean(x, (2, 3))
        return self.linear(x)


@persistence.persistent_class
class StyleEncToModspaceBlock(nn.Module):
    def __init__(self,
        imsize,
        nc1,
        nc2,
        is_first=False,
        smallest_imsize=4,
        z_dims=512,
        conv_clamp=None,
        use_fp16=False,
    ):
        super().__init__()

        # main flow
        if is_first:
            self.main = StyleEncLayer(
                nc1,
                nc2,
                z_dims=z_dims,
                down=1,
                conv_clamp=conv_clamp,
            )
        else:
            self.main = StyleResDownBlock(
                nc1,
                nc2,
                z_dims=z_dims,
                conv_clamp=conv_clamp,
                use_fp16=use_fp16,
            )

        # side flow
        nc3 = min(z_dims, nc2 * 2)
        self.side_all = StyleEncLayer(
            nc2,
            nc3,
            z_dims=z_dims,
            down=2 if log2_diff(imsize, smallest_imsize) > 1 else 1,
            conv_clamp=conv_clamp,
        )
        self.side1 = FeatMapToLatent(
            nc3,
            z_dims=z_dims,
            down=2 if log2_diff(imsize, smallest_imsize) > 0 else 1,
        )
        self.side2 = FeatMapToLatent(
            nc3,
            z_dims=z_dims,
            down=2 if log2_diff(imsize, smallest_imsize) > 0 else 1,
        )

    def forward(self, x, z):
        x = self.main(x, z)
        y = self.side_all(x, z)
        mod1 = z + self.side1(y)
        mod2 = z + self.side2(y)
        return x, mod1, mod2


@persistence.persistent_class
class StyleEncoderToModspace(nn.Module):
    def __init__(self,
        input_imsize=128,
        smallest_imsize=4,
        z_dims=512,
        nc_base=32,
        nc_max=512,
        conv_clamp=None,
        num_fp16_res=0,
    ):
        super().__init__()
        assert num_fp16_res == 0, 'TODO'
        nc_in = 3
        n_down = log2_diff(input_imsize, smallest_imsize)

        blocks = [StyleEncToModspaceBlock(
            input_imsize,
            nc_in,
            nc_base,
            is_first=True,
            smallest_imsize=smallest_imsize,
            z_dims=z_dims,
            conv_clamp=conv_clamp,
            use_fp16=False,
        )]
        imsize = input_imsize
        for i in range(n_down):
            imsize //= 2
            blocks.append(StyleEncToModspaceBlock(
                imsize,
                min(nc_max, nc_base * 2 ** i),
                min(nc_max, nc_base * 2 ** (i + 1)),
                is_first=False,
                smallest_imsize=smallest_imsize,
                z_dims=z_dims,
                conv_clamp=conv_clamp,
                use_fp16=False,
            ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, img, z):
        x = img
        mods = []
        for block in self.blocks:
            x, mod1, mod2 = block(x, z)
            mods.append(mod2)
            mods.append(mod1)
        mods = mods[::-1]
        return torch.stack(mods, dim=1)


@persistence.persistent_class
class StyleEncLayer(torch.nn.Module):
    def __init__(self,
        nc1,
        nc2,
        z_dims,
        resolution,
        kernel_size=3,
        down=2,
        activation='lrelu',
        resample_filter=[1,3,3,1],
        conv_clamp=None,
        channels_last=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.down = down
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer(
            'resample_filter',
            upfirdn2d.setup_filter(resample_filter),
        )
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(z_dims, nc1, bias_init=1)
        memory_format = torch.channels_last if channels_last else \
            torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([
            nc2,
            nc1,
            kernel_size,
            kernel_size,
        ]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([nc2]))

    def forward(self, x, z, fused_modconv=True, gain=1):
        in_res = self.resolution * self.down
        misc.assert_shape(x, [None, self.weight.shape[1], in_res, in_res])
        styles = self.affine(z)

        x = modulated_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
            up=1,
            down=self.down,
            padding=self.padding,
            resample_filter=self.resample_filter,
            flip_weight=False,
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
        )
        return x


@persistence.persistent_class
class StyleEncBlock(torch.nn.Module):
    def __init__(self,
        nc1,
        nc2,
        z_dims,
        resolution,
        resample_filter=[1,3,3,1],
        conv_clamp=None,
        use_fp16=False,
        fp16_channels_last=False,
    ):
        super().__init__()
        self.nc1 = nc1
        self.z_dims = z_dims
        self.resolution = resolution
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer(
            'resample_filter',
            upfirdn2d.setup_filter(resample_filter),
        )

        self.conv0 = StyleEncLayer(
            nc1,
            nc2,
            z_dims=z_dims,
            resolution=resolution,
            down=2,
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
        )

        self.conv1 = StyleEncLayer(
            nc2,
            nc2,
            z_dims=z_dims,
            resolution=resolution,
            down=1,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
        )

        self.skip = Conv2dLayer(
            nc1,
            nc2,
            kernel_size=1,
            bias=False,
            up=1,
            down=2,
            resample_filter=resample_filter,
            channels_last=self.channels_last,
        )

    def forward(self, x, z, force_fp32=False, fused_modconv=None):
        misc.assert_shape(z, [None, self.z_dims])
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
            self.nc1,
            self.resolution * 2,
            self.resolution * 2,
        ])
        x = x.to(dtype=dtype, memory_format=memory_format)

        y = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv0(x, z, fused_modconv=fused_modconv)
        x = self.conv1(x, z, fused_modconv=fused_modconv, gain=np.sqrt(0.5))
        x = y.add_(x)

        assert x.dtype == dtype
        return x


@persistence.persistent_class
class StyleEncoder(nn.Module):
    def __init__(self,
        imsize=128,
        smallest_imsize=4,
        z_dims=512,
        nc_in=3,
        nc_base=64,
        nc_max=512,
        to_z=True,
    ):
        super().__init__()
        n_down_up = log2_diff(imsize, smallest_imsize)
        nc = [min(nc_max, nc_base * (2 ** i)) for i in range(n_down_up + 1)]

        self.initial = Conv2dLayer(
            nc_in,
            nc[0],
            kernel_size=3,
            activation='lrelu',
        )

        blocks = []
        for i in range(n_down_up):
            blocks.append(StyleEncBlock(
                nc[i],
                nc[i+1],
                z_dims=z_dims,
                resolution=imsize // (2 ** (i + 1)),
                # use_fp16=(res >= fp16_resolution),
                use_fp16=False,
            ))
        self.blocks = nn.ModuleList(blocks)

        self.to_z = None
        if to_z:
            self.to_z = FeatMapToLatent(
                nc[-1],
                z_dims=z_dims,
                down=1,
            )

    def forward(self, img, z):
        x = self.initial(img)
        for block in self.blocks:
            x = block(x, z)
        if self.to_z is not None:
            return self.to_z(x)
        return x
