#!/usr/bin/env python3
import torch
import torch.nn as nn
from ai_old.nn.blocks.conv import EqualConv
from ai_old.nn.blocks.etc import get_actv
from ai_old.nn.blocks.norm import get_norm
from ai_old.nn.blocks.conv import ConvBlock, SmartDoubleConvBlock, DoubleConvBlock, \
    CustomConvBlock, CustomSmartDoubleConvBlock, CustomDoubleConvBlock
from ai_old.nn.blocks.se import SEModule


class FancyResConvBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        s=1,
        norm='batch',
        weight_norm=False,
        actv='swish',
        conv_clamp=None,
    ):
        super().__init__()

        if nc1 == nc2:
            self.shortcut = nn.MaxPool2d(1, s)
        else:
            if conv_clamp is not None:
                self.shortcut = CustomConvBlock(
                    nc1,
                    nc2,
                    k=1,
                    s=s,
                    norm=norm,
                    weight_norm=weight_norm,
                    actv='none',
                    conv_clamp=conv_clamp,
                )
            else:
                self.shortcut = ConvBlock(
                    nc1,
                    nc2,
                    k=1,
                    s=s,
                    norm=norm,
                    weight_norm=weight_norm,
                    actv='none',
                )

        if conv_clamp is not None:
            double = CustomSmartDoubleConvBlock(
                nc1,
                nc2,
                k=3,
                s=s,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
                conv_clamp=conv_clamp,
            )
        else:
            double = SmartDoubleConvBlock(
                nc1,
                nc2,
                k=3,
                s=s,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            )
        self.main = nn.Sequential(
            double,
            SEModule(nc2, 16, conv_clamp=conv_clamp),
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.main(x)
        return out + shortcut


class FancyMultiLayerDownBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        n_layers=2,
        norm='batch',
        weight_norm=False,
        actv='swish',
        conv_clamp=None,
    ):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(FancyResConvBlock(
                nc1 if i == 0 else nc2,
                nc2,
                s=1 if i > 0 else 2,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
                conv_clamp=conv_clamp,
            ))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResUpConvBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        k_shortcut=3,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()

        self.shortcut = ConvBlock(
            nc1,
            nc2,
            k=k_shortcut,
            norm=norm,
            weight_norm=weight_norm,
            actv='none',
        )

        self.main = DoubleConvBlock(
            nc1,
            nc1,
            nc2,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

        self.up = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True,
        )

    def forward(self, input):
        upsampled = self.up(input)
        return self.main(upsampled) + self.shortcut(upsampled)


class NoiseResUpConvBlock(nn.Module):
    def __init__(self,
        imsize,
        nc1,
        nc2,
        k_shortcut=3,
        norm='batch',
        weight_norm=False,
        actv='mish',
        conv_clamp=None,
        onnx=False,
    ):
        super().__init__()
        self.imsize = imsize

        if conv_clamp is not None:
            self.shortcut = CustomConvBlock(
                nc1,
                nc2,
                k=k_shortcut,
                norm=norm,
                weight_norm=weight_norm,
                actv='none',
                conv_clamp=conv_clamp,
                onnx=onnx,
            )
        else:
            self.shortcut = ConvBlock(
                nc1,
                nc2,
                k=k_shortcut,
                norm=norm,
                weight_norm=weight_norm,
                actv='none',
            )

        if conv_clamp is not None:
            self.main = CustomDoubleConvBlock(
                nc1,
                nc1,
                nc2,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
                conv_clamp=conv_clamp,
                onnx=onnx,
            )
        else:
            self.main = DoubleConvBlock(
                nc1,
                nc1,
                nc2,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            )

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

    def forward(self, input, noise_mode='random'):
        upsampled = self.up(input)
        x = self.main(upsampled)
        if noise_mode == 'random':
            noise = torch.randn(
                [x.shape[0], 1, self.imsize, self.imsize],
                device=x.device,
            ) * self.noise_strength
            x = x + noise
        elif noise_mode == 'const':
            noise = self.noise_const * self.noise_strength
            x = x + noise
        return x + self.shortcut(upsampled)


class ResDownConvBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        k_down=4,
        norm='batch',
        weight_norm=False,
        actv='mish',
        use_blur=False,
    ):
        super().__init__()
        assert use_blur == False

        # main flow
        main_layers = []
        main_layers.append(ConvBlock(
            nc1,
            nc2,
            k=k_down,
            s=2,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        ))
        main_layers.append(ConvBlock(
            nc2,
            nc2,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        ))
        self.main = nn.Sequential(*main_layers)

        # residual-like shortcut
        shortcut_layers = []
        shortcut_layers.append(nn.AvgPool2d(2))
        shortcut_layers.append(ConvBlock(
            nc1,
            nc2,
            k=1,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        ))
        self.shortcut = nn.Sequential(*shortcut_layers)

    def forward(self, x):
        return self.main(x) + self.shortcut(x)


class ClampResDownConvBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        k_down=4,
        norm='batch',
        weight_norm=False,
        actv='mish',
        use_blur=False,
        conv_clamp=256,
    ):
        super().__init__()
        assert use_blur == False
        assert conv_clamp is not None

        # main flow
        self.main = nn.Sequential(
            CustomConvBlock(
                nc1,
                nc2,
                k=k_down,
                s=2,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
                conv_clamp=conv_clamp,
            ),
            CustomConvBlock(
                nc2,
                nc2,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
                conv_clamp=conv_clamp,
            ),
        )

        # residual-like shortcut
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            CustomConvBlock(
                nc1,
                nc2,
                k=1,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
                conv_clamp=conv_clamp,
            )
        )

    def forward(self, x):
        return self.main(x) + self.shortcut(x)


class ResBlocks(nn.Module):
    def __init__(self,
        nc,
        n_res,
        norm='batch',
        weight_norm=False,
        actv='mish',
        pad_type='reflect',
    ):
        super().__init__()
        layers = []
        for _ in range(n_res):
            layers.append(ResBlock(
                nc,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
                pad_type=pad_type,
            ))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self,
        nc, # channels
        norm='batch',
        weight_norm=False,
        actv='mish',
        pad_type='reflect', # zeros, reflect, replicate, circular
    ):
        super().__init__()
        norm, conv_bias = get_norm(norm)
        conv = EqualConv if weight_norm else nn.Conv2d
        k = 3
        pad = (k - 1) // 2
        self.net = nn.Sequential(
            conv(
                nc, nc,
                kernel_size=3,
                padding=pad,
                bias=conv_bias,
            ),
            norm(nc),
            get_actv(actv),
            conv(
                nc, nc,
                kernel_size=3,
                padding=pad,
                bias=conv_bias,
            ),
            norm(nc),
        )

    def forward(self, x):
        y = self.net(x)
        out = x + y
        return out
