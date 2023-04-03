#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_old.nn.blocks.conv import ConvBlock, CustomConvBlock
from ai_old.nn.blocks.sg2 import Conv2DMod
from ai_old.nn.blocks.etc import get_actv


# ~~~~~~~~~~~~~~~~~
# GLOBAL MODULATION
# ~~~~~~~~~~~~~~~~~
class SleModulation(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        k=4,
    ):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d((k, k))
        self.max_pool = nn.AdaptiveMaxPool2d((k, k))

        self.net = nn.Sequential(
            nn.Conv2d(
                nc1 * 2, # *2 because avg/max concat
                nc1,
                kernel_size=k,
                stride=1,
                padding=0,
            ),
            nn.LeakyReLU(0.1),
            nn.Conv2d(nc1, nc2, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x, input):
        pooled_avg = self.avg_pool(input)
        pooled_max = self.max_pool(input)
        modulation = self.net(torch.cat((pooled_max, pooled_avg), dim=1))
        out = x * modulation # channel-wise modulation
        return out


class UpDoubleExcitationBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        z_dims,
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

        self.main1 = ExcitationBlock(
            nc1,
            nc1,
            z_dims,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

        self.main2 = ExcitationBlock(
            nc1,
            nc2,
            z_dims,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

        self.up = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True,
        )

    def forward(self, input, z):
        upsampled = self.up(input)
        out = self.main1(upsampled, z)
        out = self.main2(out, z)
        return out + self.shortcut(upsampled)


class DoubleExcitationBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        z_dims,
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

        self.main1 = ExcitationBlock(
            nc1,
            nc2,
            z_dims,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

        self.main2 = ExcitationBlock(
            nc2,
            nc2,
            z_dims,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

    def forward(self, input, z):
        out = self.main1(input, z)
        out = self.main2(out, z)
        return out + self.shortcut(input)


class DownDoubleExcitationBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        z_dims,
        k_shortcut=3,
        norm='batch',
        weight_norm=False,
        actv='mish',
        use_blur=False,
    ):
        super().__init__()

        # main flow
        self.main1 = ExcitationBlock(
            nc1,
            nc2,
            z_dims,
            s=2,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )
        self.main2 = ExcitationBlock(
            nc2,
            nc2,
            z_dims,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

        # residual-like shortcut
        shortcut_layers = [Blur()] if use_blur else []
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

    def forward(self, x, z):
        out = self.main1(x, z)
        out = self.main2(out, z)
        return out + self.shortcut(x)


class DownDoubleStyleBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        z_dims,
        k_shortcut=3,
        norm='batch',
        weight_norm=False,
        actv='mish',
        use_blur=False,
    ):
        super().__init__()
        # downsample
        down_layers = [Blur()] if use_blur else []
        down_layers.append(nn.AvgPool2d(2))
        self.down = nn.Sequential(*down_layers)

        # main flow
        self.main1 = StyleBlock(
            nc1,
            nc2,
            z_dims,
            # s=2,
            actv=actv,
        )
        self.main2 = StyleBlock(
            nc2,
            nc2,
            z_dims,
            actv=actv,
        )

        # residual-like shortcut
        # shortcut_layers = [Blur()] if use_blur else []
        # shortcut_layers.append(nn.AvgPool2d(2))
        # shortcut_layers.append(ConvBlock(
        #     nc1,
        #     nc2,
        #     k=1,
        #     norm=norm,
        #     weight_norm=weight_norm,
        #     actv=actv,
        # ))
        # self.shortcut = nn.Sequential(*shortcut_layers)
        self.shortcut = ConvBlock(
            nc1,
            nc2,
            k=1,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

    def forward(self, x, z):
        downsampled = self.down(x)
        out = self.main1(downsampled, z)
        out = self.main2(out, z)
        return out + self.shortcut(downsampled)


class UpDoubleConvertibleStyleBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        z_dims,
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

        self.main1 = ConvertibleStyleBlock(
            nc1,
            nc1,
            z_dims,
            actv=actv,
        )

        self.main2 = ConvertibleStyleBlock(
            nc1,
            nc2,
            z_dims,
            actv=actv,
        )

        self.up = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True,
        )

    def forward(self, input, z):
        upsampled = self.up(input)
        out = self.main1(upsampled, z)
        out = self.main2(out, z)
        return out + self.shortcut(upsampled)


class DoubleConvertibleStyleBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        z_dims,
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

        self.main1 = ConvertibleStyleBlock(
            nc1,
            nc2,
            z_dims,
            actv=actv,
        )

        self.main2 = ConvertibleStyleBlock(
            nc2,
            nc2,
            z_dims,
            actv=actv,
        )

    def forward(self, input, z):
        out = self.main1(input, z)
        out = self.main2(out, z)
        return out + self.shortcut(input)


# TODO: adalin

# TODO: simple adain

# TODO: global context (https://arxiv.org/pdf/2012.13375.pdf)


# ~~~~~~~~~~~~~~~~~
# LOCAL MODULATION
# ~~~~~~~~~~~~~~~~~

# TODO: spatial sle

# TODO: simple spade


# ~~~~~~~~~~~~~~~~~
# SUBMODULES
# ~~~~~~~~~~~~~~~~~
class ConvertibleStyleBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        z_dims,
        k=3,
        actv='mish',
    ):
        super().__init__()
        self.nc2 = nc2
        self.k = k

        self.to_style = nn.Linear(z_dims, nc1)

        self.weight = nn.Parameter(torch.randn((nc2, nc1, k, k)))
        nn.init.kaiming_normal_(
            self.weight,
            a=0,
            mode='fan_in',
            nonlinearity='leaky_relu',
        )

        self.pad = (k - 1) // 2
        self.actv = get_actv(actv)

    def forward(self, x, z):
        b, c, h, w = x.shape

        y = self.to_style(z)

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        d = torch.rsqrt(
            (weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + 1e-8)
        weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.nc2, *ws)

        x = F.conv2d(x, weights, padding=self.pad, groups=b)

        x = x.reshape(-1, self.nc2, h, w)
        return self.actv(x)


class StyleBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        z_dims,
        s=1,
        actv='mish',
    ):
        super().__init__()

        self.conv = Conv2DMod(
            nc1,
            nc2,
            3,
            stride=s,
        )
        self.actv = get_actv(actv)
        self.to_style = nn.Linear(z_dims, nc1)

    def forward(self, x, z):
        style = self.to_style(z)
        out = self.conv(x, style)
        return self.actv(out)


class ExcitationBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        z_dims,
        s=1,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()

        self.conv = ConvBlock(
            nc1,
            nc2,
            s=s,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

        self.excitation = nn.Sequential(
            nn.Linear(z_dims, nc2),
            nn.Sigmoid(),
        )

    def forward(self, x, z):
        excitation = self.excitation(z).unsqueeze(2).unsqueeze(3)
        return self.conv(x) * excitation

#
# class ExcitationBlockV2(nn.Module):
#     def __init__(self,
#         nc1,
#         nc2,
#         z_dims,
#         s=1,
#         norm='batch',
#         actv='mish',
#         onnx=False,
#     ):
#         super().__init__()
#
#         self.conv = CustomConvBlock(
#             nc1,
#             nc2,
#             s=s,
#             norm=norm,
#             actv=actv,
#             onnx=onnx,
#         )
#
#         self.excitation = nn.Sequential(
#             nn.Linear(z_dims, nc2),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x, z):
#         excitation = self.excitation(z).unsqueeze(2).unsqueeze(3)
#         return self.conv(x) * excitation
