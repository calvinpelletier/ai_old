#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from external.sg2 import persistence
from ai_old.util.etc import log2_diff
from ai_old.nn.blocks.conv import ConvBlock
from external.sg2.unit import FullyConnectedLayer, Conv2dLayer
from ai_old.nn.blocks.res import FancyMultiLayerDownBlock, SmartDoubleConvBlock


def _upsample_add(x, y):
    _, _, h, w = y.size()
    resized = F.interpolate(
        x,
        size=(h, w),
        mode='bilinear',
        align_corners=True,
    )
    return resized + y


@persistence.persistent_class
class FeatMapToLatent(nn.Module):
    def __init__(self,
        imsize,
        nc_in,
        z_dims,
        norm='batch',
        actv='mish',
    ):
        super().__init__()
        self.z_dims = z_dims

        max_n_down = log2_diff(imsize, 1)
        n_down = min(
            max_n_down,
            max(2, log2_diff(z_dims, nc_in)),
        )
        self.needs_avg_pool = n_down != max_n_down

        convs = []
        nc = nc_in
        for i in range(n_down):
            next_nc = min(z_dims, nc * 2)
            convs.append(ConvBlock(
                nc,
                next_nc,
                s=2,
                norm=norm,
                actv=actv,
            ))
            nc = next_nc
        self.convs = nn.Sequential(*convs)

        self.linear = FullyConnectedLayer(
            z_dims,
            z_dims,
            activation='linear',
            lr_multiplier=1,
        )

    def forward(self, x):
        x = self.convs(x)
        if self.needs_avg_pool:
            x = torch.mean(x, (2, 3))
        else:
            x = x.view(-1, self.z_dims)
        x = self.linear(x)
        return x


class UpUnetBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        norm='batch',
        actv='mish',
    ):
        super().__init__()

        self.shortcut = Conv2dLayer(
            nc1,
            nc2,
            kernel_size=1,
            up=2,
        )

        self.prep_res = ConvBlock(
            nc2,
            nc1,
            norm=norm,
            actv=actv,
        )

        self.main = SmartDoubleConvBlock(
            nc1,
            nc2,
            norm=norm,
            actv=actv,
        )

    def forward(self, input, unet_res):
        prepped_res = self.prep_res(unet_res)
        combined = _upsample_add(input, prepped_res)
        main = self.main(combined)
        return main + self.shortcut(input)


@persistence.persistent_class
class UenetEncoder(nn.Module):
    def __init__(self,
        w_avg,
        imsize=128,
        smallest_imsize=4,
        z_dims=512,
        num_ws=12,
        nc_in=3,
        nc_base=64,
        n_layers_per_res=[2, 4, 8, 4, 2],
        norm='batch',
        actv='mish',
        fmtl_norm='batch',
    ):
        super().__init__()
        assert imsize == 128 and smallest_imsize == 4
        self.num_ws = num_ws
        n_down = log2_diff(imsize, smallest_imsize)
        nc = [min(z_dims, nc_base * (2 ** i)) for i in range(n_down + 1)]

        # sanity
        assert len(n_layers_per_res) == n_down
        assert max(nc) <= z_dims
        assert z_dims == nc[-1]
        assert nc[0] == nc_base
        assert self.num_ws == 2 * (n_down + 1)

        self.register_buffer('w_avg', w_avg)

        self.deepen = ConvBlock(
            nc_in,
            nc[0],
            norm=norm,
            actv=actv,
        )
        down_blocks = []
        up_blocks = []
        fmtls = []
        for i in range(n_down):
            down_blocks.append(FancyMultiLayerDownBlock(
                nc[i],
                nc[i+1],
                n_layers=n_layers_per_res[i],
                norm=norm,
                actv=actv,
            ))
            up_blocks.append(UpUnetBlock(
                nc[i+1],
                nc[i],
                norm=norm,
                actv=actv,
            ))
            n_fmtls = 3 if i == 0 else 2
            for _ in range(n_fmtls):
                fmtls.append(FeatMapToLatent(
                    imsize=imsize // (2 ** i),
                    nc_in=nc[i],
                    z_dims=z_dims,
                    norm=fmtl_norm,
                    actv=actv,
                ))
        fmtls.append(FeatMapToLatent(
            imsize=smallest_imsize,
            nc_in=nc[-1],
            z_dims=z_dims,
            norm=fmtl_norm,
            actv=actv,
        ))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks[::-1])
        self.fmtls = nn.ModuleList(fmtls[::-1])

    def forward(self, x, return_deviation_loss=False):
        x = self.deepen(x)

        residuals = []
        for down_block in self.down_blocks:
            residuals.append(x)
            x = down_block(x)

        w0 = self.fmtls[0](x) - self.w_avg
        ws = w0.repeat(self.num_ws, 1, 1).permute(1, 0, 2)

        for i, (res, up_block) in enumerate(zip(
            residuals[::-1],
            self.up_blocks,
        )):
            x = up_block(x, res)
            idx = 1 + i * 2
            ws[:, idx] += self.fmtls[idx](x)
            ws[:, idx + 1] += self.fmtls[idx + 1](x)
            if i == len(residuals) - 1:
                ws[:, idx + 2] += self.fmtls[idx + 2](x)

        assert not return_deviation_loss
        return ws
