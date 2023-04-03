#!/usr/bin/env python3


class DiscriminatorResBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        resolution,
        img_channels,
        first_layer_idx,
        architecture='resnet',
        activation='lrelu',
        resample_filter=[1,3,3,1],
        conv_clamp=None,
        use_fp16=False,
        fp16_channels_last=False,
        freeze_layers=0,
    ):
        super().__init__()
        self.nc1 = nc1
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer(
            'resample_filter',
            upfirdn2d.setup_filter(resample_filter),
        )

        self.conv0 = Conv2dLayer(
            nc1,
            nc1,
            kernel_size=3,
            activation=activation,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
        )

        self.conv1 = Conv2dLayer(
            nc1,
            nc2,
            kernel_size=3,
            activation=activation,
            down=2,
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
        )

        self.skip = Conv2dLayer(
            nc1,
            nc2,
            kernel_size=1,
            bias=False,
            down=2,
            resample_filter=resample_filter,
            channels_last=self.channels_last,
        )

    def forward(self, x, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else \
            torch.float32
        memory_format = torch.channels_last \
            if self.channels_last and not force_fp32 else \
            torch.contiguous_format

        misc.assert_shape(x, [
            None,
            self.nc1,
            self.resolution,
            self.resolution,
        ])
        x = x.to(dtype=dtype, memory_format=memory_format)

        y = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv0(x)
        x = self.conv1(x, gain=np.sqrt(0.5))
        x = y.add_(x)

        assert x.dtype == dtype
        return x


class ShallowDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x
