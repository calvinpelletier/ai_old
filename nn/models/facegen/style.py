#!/usr/bin/env python3
import torch
import torch.nn as nn
from ai_old.nn.models import Unit
from ai_old.nn.blocks.sg2 import image_noise, GeneratorBlock, mixed_list, noise_list, \
    StyleVectorizer, latent_to_w, styles_def_to_tensor
from ai_old.util.etc import log2_diff
from math import log2
from random import random


class StyleGenerator(Unit):
    def __init__(self,
        imsize=128,
        smallest_imsize=4,
        z_dims=512,
        nc_base=32,
    ):
        super().__init__()
        self.imsize = imsize
        assert smallest_imsize == 4
        self.rank = 0

        num_layers = int(log2(imsize) - 1)
        nc = [min(z_dims, nc_base * (2 ** i)) for i in range(num_layers)][::-1]
        nc.insert(0, nc[0])
        print(nc)

        self.learned_const = nn.Parameter(torch.randn((1, nc[0], 4, 4)))

        self.blocks = nn.ModuleList([])
        for i in range(num_layers):
            block = GeneratorBlock(
                z_dims,
                nc[i],
                nc[i+1],
                upsample=i != 0,
                upsample_rgb=i != (num_layers - 1),
                rgba=False,
            )
            self.blocks.append(block)

    def forward(self, zz):
        bs = zz[0].shape[0]
        noise = image_noise(bs, self.imsize, device=self.rank)

        x = self.learned_const.expand(bs, -1, -1, -1)
        rgb = None
        for z, block in zip(zz, self.blocks):
            x, rgb = block(x, rgb, z, noise)

        return rgb

    def init_params(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(
                    m.weight,
                    a=0,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                )

        for block in self.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)


class DynamicInitStyleGenerator(Unit):
    def __init__(self,
        imsize=128,
        smallest_imsize=4,
        z_dims=512,
        nc_base=32,
    ):
        super().__init__()
        self.imsize = imsize
        assert smallest_imsize == 4
        self.rank = 0

        num_layers = int(log2(imsize) - 1)
        nc = [min(z_dims, nc_base * (2 ** i)) for i in range(num_layers)][::-1]
        nc.insert(0, nc[0])
        print(nc)

        self.blocks = nn.ModuleList([])
        for i in range(num_layers):
            block = GeneratorBlock(
                z_dims,
                nc[i],
                nc[i+1],
                upsample=i != 0,
                upsample_rgb=i != (num_layers - 1),
                rgba=False,
            )
            self.blocks.append(block)

    def forward(self, zz, initial):
        bs = zz[0].shape[0]
        noise = image_noise(bs, self.imsize, device=self.rank)

        x = initial
        rgb = None
        for z, block in zip(zz, self.blocks):
            x, rgb = block(x, rgb, z, noise)

        return rgb

    def init_params(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(
                    m.weight,
                    a=0,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                )

        for block in self.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)


class MinimalStyleGenerator(Unit):
    def __init__(self,
        imsize=128,
        smallest_imsize=4,
        z_dims=512,
        nc_base=32,
    ):
        super().__init__()
        self.imsize = imsize
        assert smallest_imsize == 4
        self.rank = 0

        num_layers = int(log2(imsize) - 1)
        nc = [min(z_dims, nc_base * (2 ** i)) for i in range(num_layers)][::-1]
        nc.insert(0, nc[0])
        print(nc)

        self.learned_const = nn.Parameter(torch.randn((1, nc[0], 4, 4)))
        self.initial_conv = nn.Conv2d(nc[0], nc[0], 3, padding=1)

        self.blocks = nn.ModuleList([])
        for i in range(num_layers):
            block = GeneratorBlock(
                z_dims,
                nc[i],
                nc[i+1],
                upsample=i != 0,
                upsample_rgb=i != (num_layers - 1),
                rgba=False,
            )
            self.blocks.append(block)

    def forward(self, z):
        bs = z.shape[0]
        noise = image_noise(bs, self.imsize, device=self.rank)

        x = self.learned_const.expand(bs, -1, -1, -1)
        rgb = None
        for block in self.blocks:
            x, rgb = block(x, rgb, z, noise)

        return rgb

    def init_params(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(
                    m.weight,
                    a=0,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                )

        for block in self.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)



class StyleGeneratorWrapper(Unit):
    def __init__(self,
        imsize=128,
        smallest_imsize=4,
        z_dims=512,
        nc_base=32,
        n_mlp=8,
        lr_mlp=0.1,
        mixed_prob=0.9,
    ):
        super().__init__()
        self.mixed_prob = mixed_prob
        self.rank = 0
        self.num_layers = int(log2(imsize) - 1)
        self.imsize = imsize
        self.z_dims = z_dims

        self.f = StyleVectorizer(z_dims, n_mlp, lr_mul=lr_mlp)
        self.g = _InnerStyleGenerator(
            imsize=imsize,
            smallest_imsize=smallest_imsize,
            z_dims=z_dims,
            nc_base=nc_base,
        )

    def forward(self, batch_size):
        get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
        style = get_latents_fn(
            batch_size,
            self.num_layers,
            self.z_dims,
            device=self.rank,
        )
        noise = image_noise(batch_size, self.imsize, device=self.rank)
        w_space = latent_to_w(self.f, style)
        w_styles = styles_def_to_tensor(w_space)
        fake = self.g(w_styles, noise)
        return {'g_fake': fake, 'zz': w_styles}

    def init_params(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(
                    m.weight,
                    a=0,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                )

        for block in self.g.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)


class _InnerStyleGenerator(Unit):
    def __init__(self,
        imsize=128,
        smallest_imsize=4,
        z_dims=512,
        nc_base=32,
    ):
        super().__init__()
        assert smallest_imsize == 4
        num_layers = int(log2(imsize) - 1)
        nc = [min(z_dims, nc_base * (2 ** i)) for i in range(num_layers)][::-1]
        nc.insert(0, nc[0])
        print(nc)

        self.to_initial_block = nn.ConvTranspose2d(
            z_dims,
            nc[0],
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False,
        )
        self.initial_conv = nn.Conv2d(nc[0], nc[0], 3, padding=1)

        self.blocks = nn.ModuleList([])
        for i in range(num_layers):
            block = GeneratorBlock(
                z_dims,
                nc[i],
                nc[i+1],
                upsample=i != 0,
                upsample_rgb=i != (num_layers - 1),
                rgba=False,
            )
            self.blocks.append(block)

    def forward(self, zz, noise):
        avg_z = zz.mean(dim=1)[:, :, None, None]
        x = self.to_initial_block(avg_z)
        rgb = None
        zz = zz.transpose(0, 1)
        x = self.initial_conv(x)
        for z, block in zip(zz, self.blocks):
            x, rgb = block(x, rgb, z, noise)
        return rgb

    def init_params(self):
        raise Exception('implemented in StyleGeneratorWrapper')
