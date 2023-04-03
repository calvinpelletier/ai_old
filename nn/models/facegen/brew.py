#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
from external.sg2 import persistence
import external.sg2.misc as misc
from external.sg2.unit import SynthesisBlock, SynthesisLayer, FullyConnectedLayer, \
    ToRGBLayer
from ai_old.util.etc import resize_imgs
from ai_old.util.factory import build_model_from_exp
from ai_old.util.pretrained import build_pretrained_sg2
from torch.nn.utils import spectral_norm
from ai_old.util.params import init_params
from ai_old.nn.models.facegen.adalin import AdalinModulatedGenerator
from ai_old.nn.models.facegen.excitation import ExcitationModulatedGenerator
from ai_old.nn.models.facegen.tri import TriLowResSynthesisNetwork
from ai_old.nn.models.encode.brew import BrewEncoderV0, BrewEncoderV1, BrewEncoderV2


class Teacher(nn.Module):
    def __init__(self, cfg, lerp_exp='lerp/5/5', low_res_exp=None):
        super().__init__()
        self.imsize = cfg.dataset.imsize
        self.low_res_exp = low_res_exp

        if self.low_res_exp is None:
            self.g = build_pretrained_sg2(g_type='tri').synthesis
        else:
            model, low_res_cfg = build_model_from_exp(low_res_exp, 'G_ema')
            assert low_res_cfg.dataset.imsize == self.imsize
            self.g = TriLowResSynthesisNetwork(
                low_res_cfg.dataset.imsize,
                channel_base=low_res_cfg.model.G.nc_base * 1024,
                num_fp16_res=4,
                conv_clamp=256,
            )
            self.g.load_state_dict(model.net.state_dict())

        self.f = build_model_from_exp(lerp_exp, 'G', return_cfg=False).f

    def forward(self, base_w, gender, mag1, mag2, mag3):
        gender = gender.unsqueeze(1)
        w1 = self.f(base_w, gender, magnitude=mag1)
        w2 = self.f(base_w, gender, magnitude=mag2)
        w3 = self.f(base_w, gender, magnitude=mag3)
        out1, out2, out3 = self.g(w1, w2, w3)
        if self.low_res_exp is None and self.imsize != 1024:
            out1 = resize_imgs(out1, self.imsize)
            out2 = resize_imgs(out2, self.imsize)
            out3 = resize_imgs(out3, self.imsize)
        return out1, out2, out3


class NonTriTeacher(nn.Module):
    def __init__(self, cfg, lerp_exp='lerp/5/5', low_res_exp=None):
        super().__init__()
        self.imsize = cfg.dataset.imsize
        self.low_res_exp = low_res_exp

        if self.low_res_exp is None:
            self.g = build_pretrained_sg2(g_type='reg').synthesis
        else:
            model, low_res_cfg = build_model_from_exp(low_res_exp, 'G_ema')
            assert low_res_cfg.dataset.imsize == self.imsize
            self.g = model.net

        self.f = build_model_from_exp(lerp_exp, 'G', return_cfg=False).f

    def forward(self, base_w, gender, mag):
        gender = gender.unsqueeze(1)
        w = self.f(base_w, gender, magnitude=mag)
        out = self.g(w)
        if self.low_res_exp is None and self.imsize != 1024:
            out = resize_imgs(out, self.imsize)
        return out


@persistence.persistent_class
class Discriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(
                3, 64, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(
                64, 128, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(
                128, 256, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(
                256, 512, 4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )
        self.apply(init_params())

    def forward(self, x):
        return self.conv(x).mean(dim=(1,2,3))

    def prep_for_train_phase(self):
        self.requires_grad_(True)


@persistence.persistent_class
class Brew(nn.Module):
    def __init__(self,
        cfg,
        nc_base=32,
        e_type='v0',
        e_pri_type='simple',
        e_mod_type='adalin',
        e_n_layers_per_res=[2, 4, 8, 4, 2],
        g_type='sg2',
        g_mod_type='excitation',
        sg2_architecture='skip',
    ):
        super().__init__()

        if e_type == 'v0':
            self.e = BrewEncoderV0(
                cfg.dataset.imsize,
                nc_base=nc_base,
                n_layers_per_res=e_n_layers_per_res,
            )
        elif e_type == 'v1':
            self.e = BrewEncoderV1(
                cfg.dataset.imsize,
                nc_base=nc_base,
                e_pri_type=e_pri_type,
                e_mod_type=e_mod_type,
            )
        elif e_type == 'v2':
            self.e = BrewEncoderV2(
                cfg.dataset.imsize,
                nc_base=nc_base,
            )
        else:
            raise Exception(e_type)

        if g_type == 'sg2':
            self.g = BrewGenerator(
                cfg.dataset.imsize,
                nc_base=nc_base,
                architecture=sg2_architecture,
            )
        elif g_type == 'simple':
            self.g = SimpleBrewGenerator(
                cfg.dataset.imsize,
                nc_base=nc_base,
                mod_type=g_mod_type,
            )
        else:
            raise Exception(g_type)

        self.apply(init_params())

    def forward(self, img, gender, magnitude):
        idt, w, delta = self.e(img, gender)
        out = self.g(idt, w, delta, magnitude)
        return out

    def prep_for_train_phase(self):
        self.requires_grad_(True)


class BrewGenerator(nn.Module):
    def __init__(self, imsize, nc_base=32, nc_max=512, architecture='skip'):
        super().__init__()
        self.net = SynthesisNetwork(
            imsize,
            channel_base=nc_base * imsize,
            channel_max=nc_max,
            architecture=architecture,
        )
        self.num_ws = self.net.num_ws

    def forward(self, idt, w, delta, magnitude):
        misc.assert_shape(idt, [None, 512, 4, 4])
        misc.assert_shape(w, [None, 512])
        misc.assert_shape(delta, [None, 512])
        target_w = w + delta * magnitude
        w_plus = target_w.unsqueeze(1).repeat(1, self.num_ws, 1)
        return self.net(idt, w_plus, noise_mode='const')


class SimpleBrewGenerator(nn.Module):
    def __init__(self, imsize, nc_base=32, nc_max=512, mod_type='excitation'):
        super().__init__()
        if mod_type == 'adalin':
            cls = AdalinModulatedGenerator
        elif mod_type == 'excitation':
            cls = ExcitationModulatedGenerator
        else:
            raise ValueError(mod_type)
        self.net = cls(
            output_imsize=imsize,
            init_imsize=4,
            nc_in=3,
            nc_base=nc_base,
            nc_max=nc_max,
            z_dims=512,
            norm='batch',
            weight_norm=False,
            actv='mish',
        )

    def forward(self, idt, w, delta, magnitude):
        misc.assert_shape(idt, [None, 512, 4, 4])
        misc.assert_shape(w, [None, 512])
        misc.assert_shape(delta, [None, 512])
        target_w = w + delta * magnitude
        return self.net(idt, target_w)


class SynthesisNetwork(nn.Module):
    def __init__(self,
        imsize,
        w_dim=512,
        img_channels=3,
        channel_base=32768,
        channel_max=512,
        architecture='skip',
    ):
        assert imsize >= 4 and \
            imsize & (imsize - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.imsize = imsize
        self.img_resolution_log2 = int(np.log2(imsize))
        self.img_channels = img_channels
        self.block_resolutions = [
            2 ** i for i in range(2, self.img_resolution_log2 + 1)
        ]
        channels_dict = {
            res: min(channel_base // res, channel_max) \
            for res in self.block_resolutions
        }

        self.num_ws = 0
        for res in self.block_resolutions:
            is_last = (res == self.imsize)
            if res > 4:
                block = SynthesisBlock(
                    channels_dict[res // 2],
                    channels_dict[res],
                    w_dim=w_dim,
                    resolution=res,
                    img_channels=img_channels,
                    is_last=is_last,
                    architecture=architecture,
                )
            else:
                block = FirstSynthesisBlock(
                    channels_dict[res],
                    w_dim=w_dim,
                    resolution=res,
                    img_channels=img_channels,
                    architecture=architecture,
                )
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, idt, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(
                    1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        block = getattr(self, 'b4')
        x, img = block(idt, block_ws[0], **block_kwargs)

        for res, cur_ws in zip(self.block_resolutions[1:], block_ws[1:]):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)

        return img


class FirstSynthesisBlock(nn.Module):
    def __init__(self,
        out_channels,
        w_dim,
        resolution,
        img_channels,
        architecture='skip',
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture
        self.num_conv = 0
        self.num_torgb = 0

        self.conv1 = SynthesisLayer(
            out_channels,
            out_channels,
            w_dim=w_dim,
            resolution=resolution,
        )
        self.num_conv += 1

        if architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim)
            self.num_torgb += 1

    def forward(self, x, ws, **layer_kwargs):
        w_iter = iter(ws.unbind(dim=1))

        # main
        x = self.conv1(x, next(w_iter), **layer_kwargs)

        # to rgb
        if self.architecture == 'skip':
            img = self.torgb(x, next(w_iter))
        else:
            img = None

        return x, img
