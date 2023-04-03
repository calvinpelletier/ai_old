#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from external.sg2 import persistence
from ai_old.util.pretrained import build_pretrained_sg2
from ai_old.util.params import init_params
from ai_old.util.etc import resize_imgs
from external.sg2.unit import FullyConnectedLayer
from ai_old.nn.blocks.norm import PixelNorm
from ai_old.util.factory import build_model_from_exp
import external.sg2.misc as misc


class DynamicLerper(nn.Module):
    def __init__(self, final_activation, lr_mul, pixel_norm):
        super().__init__()
        layers = [PixelNorm()] if pixel_norm else []
        n_layers = 4
        for i in range(n_layers):
            actv = final_activation if i == n_layers - 1 else 'lrelu'
            # assert actv in ['linear', 'lrelu']
            # actv = actv == 'lrelu'
            layers.append(FullyConnectedLayer(
                512,
                512,
                activation=actv,
                lr_multiplier=lr_mul,
            ))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LevelsDynamicLerper(nn.Module):
    def __init__(self,
        levels,
        final_activation,
        mult,
        lr_mul,
        gendered,
        pixel_norm,
    ):
        super().__init__()
        self.gendered = gendered
        self.mult = mult
        self.coarse_enabled = 'coarse' in levels
        self.medium_enabled = 'medium' in levels
        self.fine_enabled = 'fine' in levels

        if self.coarse_enabled:
            self.coarse_lerper = DynamicLerper(
                final_activation, lr_mul, pixel_norm)
        if self.medium_enabled:
            self.medium_lerper = DynamicLerper(
                final_activation, lr_mul, pixel_norm)
        if self.fine_enabled:
            self.fine_lerper = DynamicLerper(
                final_activation, lr_mul, pixel_norm)

    def forward(self, w, gender, magnitude=1.):
        if torch.is_tensor(magnitude):
            bs = w.shape[0]
            misc.assert_shape(magnitude, [bs])
            magnitude = torch.reshape(magnitude, (bs, 1, 1))

        delta = torch.zeros_like(w)
        for i in range(18):
            if i < 4:
                if self.coarse_enabled:
                    delta[:, i, :] = self.coarse_lerper(w[:, i, :])
            elif i < 8:
                if self.medium_enabled:
                    delta[:, i, :] = self.medium_lerper(w[:, i, :])
            else:
                if self.fine_enabled:
                    delta[:, i, :] = self.fine_lerper(w[:, i, :])
        delta = delta * self.mult * magnitude
        if self.gendered:
            gender_mult = (gender * 2. - 1.).unsqueeze(dim=1)
            delta = delta * gender_mult
        return w + delta


@persistence.persistent_class
class PretrainedGenerator(nn.Module):
    def __init__(self, output_imsize):
        super().__init__()
        self.output_imsize = output_imsize

        G = build_pretrained_sg2()
        self.net = G.synthesis.to('cuda').eval()
        self.net.requires_grad_(False)

    def forward(self, w):
        img = self.net(w, noise_mode='const')
        img = resize_imgs(img, self.output_imsize)
        return img


@persistence.persistent_class
class LerperAndGenerator(nn.Module):
    def __init__(self,
        cfg,
        lerper_type='levels',
        levels=['coarse', 'medium', 'fine'],
        final_activation='linear',
        mult=1.,
        lr_mul=1.,
        pixel_norm=True,
    ):
        super().__init__()
        self.output_imsize = cfg.dataset.imsize
        gendered = hasattr(cfg, 'gendered') and cfg.gendered

        self.g = PretrainedGenerator(output_imsize=self.output_imsize)
        self.g.requires_grad_(False)

        if lerper_type == 'levels':
            self.f = LevelsDynamicLerper(
                levels=levels,
                final_activation=final_activation,
                mult=mult,
                lr_mul=lr_mul,
                gendered=gendered,
                pixel_norm=pixel_norm,
            )
        else:
            raise Exception(lerper_type)
        self.f.apply(init_params())

    def forward(self, w, gender, magnitude=1.):
        new_w = self.f(w, gender, magnitude)
        new_img = self.g(new_w)
        return new_img

    def prep_for_train_phase(self):
        self.f.requires_grad_(True)


# NOTE: gradients do not pass between the two lerpers
# class DualLevelsDynamicLerpers(nn.Module):
#     def __init__(self,
#         levels1,
#         levels2,
#         final_activation,
#         mult,
#         lr_mul,
#         gendered,
#         detach_f2,
#     ):
#         super().__init__()
#         self.detach_f2 = detach_f2
#
#         self.f1 = LevelsDynamicLerper(
#             levels=levels1,
#             final_activation=final_activation,
#             mult=mult,
#             lr_mul=lr_mul,
#             gendered=gendered,
#         )
#
#         self.f2 = LevelsDynamicLerper(
#             levels=levels2,
#             final_activation=final_activation,
#             mult=mult,
#             lr_mul=lr_mul,
#             gendered=gendered,
#         )
#
#     def forward(self, w, gender, magnitude=1.):
#         w1 = self.f1(w, gender, magnitude=magnitude)
#         if self.detach_f2:
#             f2_input = w1.clone().detach()
#         else:
#             f2_input = w1
#         w2 = self.f2(f2_input, gender, magnitude=magnitude)
#         return w1, w2


@persistence.persistent_class
class PretrainedSegGenerator(nn.Module):
    def __init__(self, output_imsize, exp='seg/0/0'):
        super().__init__()
        self.output_imsize = output_imsize

        self.g = build_pretrained_sg2(g_type='seg').synthesis.to('cuda')
        self.s = build_model_from_exp(exp, 'G', return_cfg=False).s.to('cuda')

    def forward(self, w):
        img, feats = self.g(w, noise_mode='const')
        seg = self.s(feats)
        img = resize_imgs(img, self.output_imsize)
        seg = resize_imgs(seg, self.output_imsize)
        return img, seg


@persistence.persistent_class
class DualLerperAndSegGenerator(nn.Module):
    def __init__(self,
        cfg,
        f1_exp='lerp/5/5',
        lerper_type='levels',
        levels=['coarse', 'medium', 'fine'],
        final_activation='linear',
        mult=1.,
        lr_mul=1.,
    ):
        super().__init__()
        self.output_imsize = cfg.dataset.imsize
        gendered = hasattr(cfg, 'gendered') and cfg.gendered

        self.g = PretrainedSegGenerator(output_imsize=self.output_imsize)
        self.g.eval()
        self.g.requires_grad_(False)

        self.f1 = build_model_from_exp(f1_exp, 'G', return_cfg=False).f
        self.f1.eval()
        self.f1.requires_grad_(False)

        if lerper_type == 'levels':
            self.f2 = LevelsDynamicLerper(
                levels=levels,
                final_activation=final_activation,
                mult=mult,
                lr_mul=lr_mul,
                gendered=gendered,
            )
        else:
            raise Exception(lerper_type)
        self.f2.apply(init_params())

    def forward(self, w, gender, magnitude=1.):
        w1 = self.f1(w, gender, magnitude)
        w2 = self.f2(w1, gender, magnitude)
        img1, seg1 = self.g(w1)
        img2, seg2 = self.g(w2)
        return img1, img2

    def prep_for_train_phase(self):
        self.f2.requires_grad_(True)
