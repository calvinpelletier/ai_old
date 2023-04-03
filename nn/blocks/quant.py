#!/usr/bin/env python3
import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, nc1, nc2, k=3, s=1):
        padding = (k - 1) // 2
        super().__init__(
            nn.Conv2d(nc1, nc2, k, s, padding, bias=False),
            nn.BatchNorm2d(nc2),
            nn.ReLU(inplace=False),
        )


class ConvBN(nn.Sequential):
    def __init__(self, nc1, nc2, k=3, s=1):
        padding = (k - 1) // 2
        super().__init__(
            nn.Conv2d(nc1, nc2, k, s, padding, bias=False),
            nn.BatchNorm2d(nc2),
        )


class DoubleConvBNReLU(nn.Sequential):
    def __init__(self, nc1, nc2, k=3):
        padding = (k - 1) // 2
        super().__init__(
            nn.Conv2d(nc1, nc1, k, 1, padding, bias=False),
            nn.BatchNorm2d(nc1),
            nn.ReLU(inplace=False),
            nn.Conv2d(nc1, nc2, k, 1, padding, bias=False),
            nn.BatchNorm2d(nc2),
            nn.ReLU(inplace=False),
        )


class SimpleNoiseResUpConvBlock(nn.Module):
    def __init__(self, imsize, nc1, nc2):
        super().__init__()
        self.imsize = imsize

        self.shortcut = ConvBN(nc1, nc2)
        self.main = DoubleConvBNReLU(nc1, nc2)

        self.up = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True,
        )

        self.register_buffer(
            'noise_const',
            torch.randn([imsize, imsize]),
        )
        self.noise_strength = nn.Parameter(torch.zeros([]))

        # self.skip_add = nn.quantized.FloatFunctional()
        # self.noise_add = nn.quantized.FloatFunctional()

    def forward(self, input, noise_mode='random'):
        upsampled = self.up(input)
        x = self.main(upsampled)
        if noise_mode == 'random':
            noise = torch.randn(
                [x.shape[0], 1, self.imsize, self.imsize],
                device=x.device,
            ) * self.noise_strength
            # x = self.noise_add.add(x, noise)
            x = x + noise
        elif noise_mode == 'const':
            noise = self.noise_const * self.noise_strength
            # x = self.noise_add.add(x, noise)
            x = x + noise
        # return self.skip_add.add(x, self.shortcut(upsampled))
        return x + self.shortcut(upsampled)

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(
                    m,
                    ['0', '1', '2'],
                    inplace=True,
                )
            elif type(m) == ConvBN:
                torch.quantization.fuse_modules(
                    m,
                    ['0', '1'],
                    inplace=True,
                )
            elif type(m) == DoubleConvBNReLU:
                torch.quantization.fuse_modules(
                    m,
                    ['0', '1', '2'],
                    inplace=True,
                )
                torch.quantization.fuse_modules(
                    m,
                    ['3', '4', '5'],
                    inplace=True,
                )


class QuantSimpleNoiseResUpConvBlock(nn.Module):
    def __init__(self, imsize, nc1, nc2):
        super().__init__()
        self.imsize = imsize

        self.shortcut = ConvBN(nc1, nc2)
        self.main = DoubleConvBNReLU(nc1, nc2)

        self.up = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True,
        )

        self.register_buffer(
            'noise_const',
            torch.randn([imsize, imsize]),
        )
        self.noise_strength = nn.Parameter(torch.zeros([]))
        self.noise_quant = torch.quantization.QuantStub()

        # self.skip_add = nn.quantized.FloatFunctional()
        # self.noise_add = nn.quantized.FloatFunctional()

    def forward(self, input, noise_mode='random'):
        upsampled = self.up(input)
        x = self.main(upsampled)
        if noise_mode == 'random':
            noise = torch.randn(
                [x.shape[0], 1, self.imsize, self.imsize],
                device=x.device,
            ) * self.noise_strength
            # x = self.noise_add.add(x, noise)
            x = x + noise
        elif noise_mode == 'const':
            noise = self.noise_const * self.noise_strength
            noise = self.noise_quant(noise)
            # x = self.noise_add.add(x, noise)
            x = x + noise
        # return self.skip_add.add(x, self.shortcut(upsampled))
        return x + self.shortcut(upsampled)

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(
                    m,
                    ['0', '1', '2'],
                    inplace=True,
                )
            elif type(m) == ConvBN:
                torch.quantization.fuse_modules(
                    m,
                    ['0', '1'],
                    inplace=True,
                )
            elif type(m) == DoubleConvBNReLU:
                torch.quantization.fuse_modules(
                    m,
                    ['0', '1', '2'],
                    inplace=True,
                )
                torch.quantization.fuse_modules(
                    m,
                    ['3', '4', '5'],
                    inplace=True,
                )
