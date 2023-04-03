#!/usr/bin/env python3
import torch
import torch.nn as nn


@persistence.persistent_class
class ArcfaceEncoderToModspace(nn.Module):
    def __init__(self,
        input_imsize=128,
        smallest_imsize=4,
        z_dims=512,
        nc_base=32,
        nc_max=512,
    ):
        super().__init__()
        num_layers = 50
        nc_in = 3
        
        blocks = get_blocks(num_layers)
        self.input_layer = ConvBlock(
            nc_in,
            64,
            norm='batch',
            weight_norm=False,
            activation='prelu',
        )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck_IR_SE(
                    bottleneck.in_channel,
                    bottleneck.depth,
                    bottleneck.stride,
                ))
        self.body = nn.Sequential(*modules)

        self.latents = nn.ModuleList()
        self.latent_count = 18
        self.coarse_idx = 3
        self.middle_idx = 7
        assert conf.dataset.imsize in [128, 256]
        for i in range(self.latent_count):
            if i < self.coarse_idx:
                latent = FullDown(
                    512,
                    512,
                    16 if conf.dataset.imsize == 256 else 8,
                    norm=conf.model.e.norm,
                    weight_norm=conf.model.e.weight_norm,
                    activation=conf.model.e.activation,
                )
            elif i < self.middle_idx:
                latent = FullDown(
                    512,
                    512,
                    32 if conf.dataset.imsize == 256 else 16,
                    norm=conf.model.e.norm,
                    weight_norm=conf.model.e.weight_norm,
                    activation=conf.model.e.activation,
                )
            else:
                latent = FullDown(
                    512,
                    512,
                    64 if conf.dataset.imsize == 256 else 32,
                    norm=conf.model.e.norm,
                    weight_norm=conf.model.e.weight_norm,
                    activation=conf.model.e.activation,
                )
            self.latents.append(latent)
        self.deepen1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.deepen2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def init_params(self):
        # head
        fn = init_params()
        self.input_layer.apply(fn)
        self.latents.apply(fn)
        self.deepen1.apply(fn)
        self.deepen2.apply(fn)

        # init body with arcface weights
        dep_conf = getattr(self.conf.deps, self.conf.model.e.init)
        path = os.path.join(c.PRETRAINED_MODELS, dep_conf.path)
        self.load_state_dict(torch.load(path), strict=False)

    def _up_add(self, x, y):
        _, _, h, w = y.size()
        x = F.interpolate(
            x,
            size=(h, w),
            mode='bilinear',
            align_corners=True,
        )
        return x + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, blk in enumerate(modulelist):
            x = blk(x)
            if i == 6:
                fine = x
            elif i == 20:
                middle = x
            elif i == 23:
                coarse = x

        for i in range(self.coarse_idx):
            latents.append(self.latents[i](coarse))

        coarse_middle = self._up_add(coarse, self.deepen1(middle))
        for i in range(self.coarse_idx, self.middle_idx):
            latents.append(self.latents[i](coarse_middle))

        coarse_middle_fine = self._up_add(
            coarse_middle, self.deepen2(fine))
        for i in range(self.middle_idx, self.latent_count):
            latents.append(self.latents[i](coarse_middle_fine))

        out = torch.stack(latents, dim=1)
        return out
