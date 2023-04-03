#!/usr/bin/env python3
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F
from ai_old.util.etc import log2_diff


class UpSpadeResBlocks(nn.Module):
    def __init__(self,
        imsize=128,
        smallest_imsize=8,
        nc_init=256,
        nc_last=32,
        norm='batch',
        use_spectral_norm=True,
    ):
        super().__init__()
        # hardcoding size for now
        # TODO: make configurable
        assert imsize == 128 and smallest_imsize == 8
        n_spade = log2_diff(imsize, smallest_imsize) + 1
        assert n_spade == 5
        nc = [min(nc_init, nc_last * 2**(i-1)) for i in range(n_spade, 0, -1)]
        assert nc[-1] == nc[4] == nc_last, f'{nc[-1]}, {nc[4]}, {nc_last}'
        if nc_init == 256 and nc_last == 32:
            assert nc == [256, 256, 128, 64, 32]

        self.up = nn.Upsample(scale_factor=2)

        # TODO: reduce duplicated code (can probably just do partials)
        self.spade0 = SpadeResBlock(
            nc_init,
            nc[0],
            norm=norm,
            use_spectral_norm=use_spectral_norm,
        )
        self.spade1 = SpadeResBlock(
            nc[0],
            nc[1],
            norm=norm,
            use_spectral_norm=use_spectral_norm,
        )
        self.spade2 = SpadeResBlock(
            nc[1],
            nc[2],
            norm=norm,
            use_spectral_norm=use_spectral_norm,
        )
        self.spade3 = SpadeResBlock(
            nc[2],
            nc[3],
            norm=norm,
            use_spectral_norm=use_spectral_norm,
        )
        self.spade4 = SpadeResBlock(
            nc[3],
            nc[4],
            norm=norm,
            use_spectral_norm=use_spectral_norm,
        )

    def forward(self, encoding, input):
        x = self.spade0(encoding, input)
        x = self.up(x)
        x = self.spade1(x, input)
        x = self.up(x)
        x = self.spade2(x, input)
        x = self.up(x)
        x = self.spade3(x, input)
        x = self.up(x)
        x = self.spade4(x, input)
        return x


class SpadeResBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        modulator_nc=3, # rgb
        norm='batch',
        use_spectral_norm=True,
    ):
        super().__init__()
        self.learned_shortcut = (nc1 != nc2)

        nc = min(nc1, nc2)

        self.conv0 = nn.Conv2d(nc1, nc, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(nc, nc2, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_shortcut = nn.Conv2d(nc1, nc2, kernel_size=1, bias=False)

        if use_spectral_norm:
            self.conv0 = spectral_norm(self.conv0)
            self.conv1 = spectral_norm(self.conv1)
            if self.learned_shortcut:
                self.conv_shortcut = spectral_norm(self.conv_shortcut)

        self.spade0 = Spade(nc1, modulator_nc, norm=norm)
        self.spade1 = Spade(nc, modulator_nc, norm=norm)
        if self.learned_shortcut:
            self.spade_shortcut = Spade(nc1, modulator_nc, norm=norm)

    def forward(self, x, input):
        if self.learned_shortcut:
            x_s = self.conv_shortcut(self.spade_shortcut(x, input))
        else:
            x_s = x

        dx = self.conv0(self.act(self.spade0(x, input)))
        dx = self.conv1(self.act(self.spade1(dx, input)))
        return x_s + dx

    def act(self, x):
        return F.leaky_relu(x, 2e-1)


class Spade(nn.Module):
    def __init__(self, nc, nc_in, norm='batch'):
        super().__init__()
        k = 3
        pad = k // 2
        hidden = 128

        if norm == 'instance':
            self.norm = nn.InstanceNorm2d(nc, affine=False)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(nc, affine=False)
        else:
            raise Exception('unimplemented norm: ' + conf.model.g.norm)

        self.shared = nn.Sequential(
            nn.Conv2d(nc_in, hidden, kernel_size=k, padding=pad),
            nn.ReLU(),
        )
        self.gamma = nn.Conv2d(hidden, nc, kernel_size=k, padding=pad)
        self.beta = nn.Conv2d(hidden, nc, kernel_size=k, padding=pad)

    def forward(self, x, input):
        normed = self.norm(x)
        input = F.interpolate(input, size=x.size()[2:], mode='bilinear')
        shared = self.shared(input)
        gamma = self.gamma(shared)
        beta = self.beta(shared)
        return normed * (1 + gamma) + beta
