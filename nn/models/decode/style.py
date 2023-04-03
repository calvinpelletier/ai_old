#!/usr/bin/env python3
import torch
import torch.nn as nn
from external.sg2 import persistence
from external.sg2.unit import SynthesisLayer, Conv2dLayer, ToRGBLayer
import external.sg2.misc as misc
from external.op import upfirdn2d
import numpy as np
from ai_old.nn.models.decode.l2fm import LearnedHybridLatentToFeatMap


@persistence.persistent_class
class StyleDecoderBlock(torch.nn.Module):
    def __init__(self,
        nc1,
        nc2,
        z_dims,
        resolution,
        nc_in,
        dont_upsample,
        is_last,
        architecture='skip',
        resample_filter=[1,3,3,1],
        conv_clamp=None,
        use_fp16=False,
        fp16_channels_last=False,
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        assert nc1 != 0, 'no support for learned const start block'
        super().__init__()
        self.nc1 = nc1
        self.z_dims = z_dims
        self.resolution = resolution
        self.nc_in = nc_in
        self.dont_upsample = dont_upsample
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer(
            'resample_filter',
            upfirdn2d.setup_filter(resample_filter),
        )

        self.conv0 = SynthesisLayer(
            nc1,
            nc2,
            w_dim=z_dims,
            resolution=resolution,
            up=1 if dont_upsample else 2,
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
        )

        self.conv1 = SynthesisLayer(
            nc2,
            nc2,
            w_dim=z_dims,
            resolution=resolution,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
        )

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(nc2, nc_in, w_dim=z_dims,
                conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(
                nc1,
                nc2,
                kernel_size=1,
                bias=False,
                up=1 if dont_upsample else 2,
                resample_filter=resample_filter,
                channels_last=self.channels_last,
            )

    def forward(self,
        x,
        img,
        z,
        force_fp32=False,
        fused_modconv=None,
    ):
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
            self.resolution if self.dont_upsample else self.resolution // 2,
            self.resolution if self.dont_upsample else self.resolution // 2,
        ])
        x = x.to(dtype=dtype, memory_format=memory_format)

        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, z, fused_modconv=fused_modconv)
            x = self.conv1(x, z, fused_modconv=fused_modconv, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x, z, fused_modconv=fused_modconv)
            x = self.conv1(x, z, fused_modconv=fused_modconv)

        if img is not None:
            misc.assert_shape(img, [
                None,
                self.nc_in,
                self.resolution if self.dont_upsample else self.resolution // 2,
                self.resolution if self.dont_upsample else self.resolution // 2,
            ])
            if not self.dont_upsample:
                img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, z, fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img


@persistence.persistent_class
class StyleDecoder(nn.Module):
    def __init__(self,
        imsize=128,
        smallest_imsize=4,
        z_dims=512,
        nc_in=3,
        nc_base=64,
        nc_max=512,
        from_z_only=False,
    ):
        super().__init__()
        assert imsize >= 4 and imsize & (imsize - 1) == 0
        assert smallest_imsize == 4
        self.from_z_only = from_z_only
        self.z_dims = z_dims
        self.imsize = imsize
        self.log2_imsize = int(np.log2(imsize))
        self.nc_in = nc_in
        self.block_resolutions = [
            2 ** i for i in range(2, self.log2_imsize + 1)
        ]
        channels_dict = {
            res: min((nc_base * imsize) // res, nc_max) \
            for res in self.block_resolutions
        }
        # fp16_resolution = max(
        #     2 ** (self.log2_imsize + 1 - num_fp16_res),
        #     8,
        # )

        for res in self.block_resolutions:
            block = StyleDecoderBlock(
                channels_dict[res // 2] if res > 4 else channels_dict[res],
                channels_dict[res],
                z_dims=z_dims,
                resolution=res,
                nc_in=nc_in,
                dont_upsample=(res == 4),
                is_last=(res == self.imsize),
                # use_fp16=(res >= fp16_resolution),
                use_fp16=False,
            )
            setattr(self, f'b{res}', block)

        if self.from_z_only:
            self.z_to_feat_map = LearnedHybridLatentToFeatMap(z_dims)

    def forward(self, x, z):
        if self.from_z_only:
            x = self.z_to_feat_map(z)

        img = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, z)
        return img


@persistence.persistent_class
class ZOnlyStyleDecoder(nn.Module):
    def __init__(self,
        imsize=128,
        smallest_imsize=4,
        z_dims=512,
        nc_in=3,
        nc_base=64,
        nc_max=512,
        from_z_only=False,
    ):
        super().__init__()
        assert imsize >= 4 and imsize & (imsize - 1) == 0
        assert smallest_imsize == 4
        self.z_dims = z_dims
        self.imsize = imsize
        self.log2_imsize = int(np.log2(imsize))
        self.nc_in = nc_in
        self.block_resolutions = [
            2 ** i for i in range(2, self.log2_imsize + 1)
        ]
        channels_dict = {
            res: min((nc_base * imsize) // res, nc_max) \
            for res in self.block_resolutions
        }
        # fp16_resolution = max(
        #     2 ** (self.log2_imsize + 1 - num_fp16_res),
        #     8,
        # )

        for res in self.block_resolutions:
            block = StyleDecoderBlock(
                channels_dict[res // 2] if res > 4 else channels_dict[res],
                channels_dict[res],
                z_dims=z_dims,
                resolution=res,
                nc_in=nc_in,
                dont_upsample=(res == 4),
                is_last=(res == self.imsize),
                # use_fp16=(res >= fp16_resolution),
                use_fp16=False,
            )
            setattr(self, f'b{res}', block)

        self.z_to_feat_map = LearnedHybridLatentToFeatMap(z_dims)

    def forward(self, z):
        x = self.z_to_feat_map(z)
        img = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, z)
        return img


@persistence.persistent_class
class LearnedConstStyleDecoder(nn.Module):
    def __init__(self,
        imsize=128,
        smallest_imsize=4,
        z_dims=512,
        nc_in=3,
        nc_base=64,
        nc_max=512,
        additional_4x4s=0,
    ):
        super().__init__()
        assert imsize >= 4 and imsize & (imsize - 1) == 0
        assert smallest_imsize == 4
        self.z_dims = z_dims
        self.imsize = imsize
        self.log2_imsize = int(np.log2(imsize))
        self.nc_in = nc_in

        self.block_resolutions = [4 for _ in range(additional_4x4s)]
        for i in range(2, self.log2_imsize + 1):
            self.block_resolutions.append(2 ** i)

        channels_dict = {
            res: min((nc_base * imsize) // res, nc_max) \
            for res in self.block_resolutions
        }
        # fp16_resolution = max(
        #     2 ** (self.log2_imsize + 1 - num_fp16_res),
        #     8,
        # )

        for i, res in enumerate(self.block_resolutions):
            block = StyleDecoderBlock(
                channels_dict[res // 2] if res > 4 else channels_dict[res],
                channels_dict[res],
                z_dims=z_dims,
                resolution=res,
                nc_in=nc_in,
                dont_upsample=(res == 4),
                is_last=(res == self.imsize),
                # use_fp16=(res >= fp16_resolution),
                use_fp16=False,
            )
            setattr(self, f'b{res}_{i}', block)

        self.const = nn.Parameter(torch.randn([
            z_dims,
            smallest_imsize,
            smallest_imsize,
        ]))

    def forward(self, z):
        x = self.const.unsqueeze(0).repeat([z.shape[0], 1, 1, 1])
        img = None
        for i, res in enumerate(self.block_resolutions):
            block = getattr(self, f'b{res}_{i}')
            x, img = block(x, img, z)
        return img
