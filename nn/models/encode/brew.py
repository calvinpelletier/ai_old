#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_old.nn.blocks.res import FancyMultiLayerDownBlock, FancyResConvBlock
from ai_old.nn.models.encode.squeeze import Encoder
from ai_old.nn.models.encode.fm2l import SimpleFeatMapToLatent
from external.sg2.unit import FullyConnectedLayer
from ai_old.nn.models.encode.simple import SimpleZEncoder
from ai_old.nn.models.encode.adalin import AdalinModulatedEncoder
from ai_old.nn.models.encode.excitation import ExcitationModulatedEncoder
from ai_old.nn.models.encode.style import StyleEncoder
from ai_old.nn.models.encode.squeeze import ZEncoder


class FcLayers(nn.Module):
    def __init__(self, n_layers, lr_mul):
        super().__init__()
        layers = []
        for i in range(n_layers):
            actv = 'linear' if i == n_layers - 1 else 'lrelu'
            layers.append(FullyConnectedLayer(
                512,
                512,
                activation=actv,
                lr_multiplier=lr_mul,
            ))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BrewEncoderV0(nn.Module):
    def __init__(self,
        imsize,
        nc_base=32,
        n_layers_per_res=[2, 4, 8, 4, 2],
        norm='batch',
        actv='mish',
    ):
        super().__init__()
        self.gender_enc_dims = 16
        self.inner_res = 8

        self.img_encoder = Encoder(
            input_imsize=imsize,
            smallest_imsize=self.inner_res,
            nc_base=nc_base,
            nc_max=512,
            n_layers_per_res=n_layers_per_res[:-1],
            norm=norm,
            actv=actv,
        )

        self.gender_encoder = FullyConnectedLayer(1, self.gender_enc_dims)

        self.gender_img_mixer = FancyResConvBlock(
            512 + self.gender_enc_dims,
            512,
            s=1,
            norm=norm,
            actv=actv,
        )

        self.idt_encoder = FancyMultiLayerDownBlock(
            512,
            512,
            n_layers=n_layers_per_res[-1],
            norm=norm,
            actv=actv,
        )

        self.w_encoder = SimpleFeatMapToLatent(self.inner_res, 512, 512)
        self.delta_encoder = SimpleFeatMapToLatent(self.inner_res, 512, 512)

    def forward(self, img, gender):
        assert len(gender.shape) == 1
        gender_enc = self.gender_encoder(gender.unsqueeze(1))
        gender_enc = torch.reshape(gender_enc, (-1, self.gender_enc_dims, 1, 1))
        gender_enc = gender_enc.repeat(1, 1, self.inner_res, self.inner_res)

        img_enc = self.img_encoder(img)
        x = self.gender_img_mixer(torch.cat([img_enc, gender_enc], dim=1))

        idt = self.idt_encoder(x)
        w = self.w_encoder(x)
        delta = self.delta_encoder(x)

        return idt, w, delta


class BrewEncoderV1(nn.Module):
    def __init__(self,
        imsize,
        nc_base=32,
        e_pri_type='simple',
        e_mod_type='adalin',
    ):
        super().__init__()
        self.imsize = imsize

        assert e_pri_type == 'simple'
        self.e_pri = SimpleZEncoder(
            input_imsize=imsize,
            smallest_imsize=4,
            z_dims=512,
            k=3,
            k_init=3,
            nc_in=4,
            nc_base=nc_base,
            im2vec='mlp',
            norm='batch',
            weight_norm=False,
            actv='mish',
        )

        if e_mod_type == 'adalin':
            e_mod_cls = AdalinModulatedEncoder
        elif e_mod_type == 'excitation':
            e_mod_cls = ExcitationModulatedEncoder
        else:
            raise ValueError(e_mod)
        self.e_mod = e_mod_cls(
            input_imsize=imsize,
            smallest_imsize=4,
            nc_in=3,
            nc_base=nc_base,
            nc_max=512,
            z_dims=512,
            norm='batch',
            weight_norm=False,
            actv='mish',
        )

        self.f = FcLayers(4, 1.)

    def forward(self, img, gender):
        assert len(gender.shape) == 1
        gender = torch.reshape(gender, (-1, 1, 1, 1))
        gender = gender.repeat(1, 1, self.imsize, self.imsize)

        z = self.e_pri(torch.cat([img, gender], dim=1))
        idt = self.e_mod(img, z)
        delta = self.f(z)
        return idt, z, delta


class BrewEncoderV2(nn.Module):
    def __init__(self,
        imsize,
        nc_base=32,
        inst_norm_idt=True,
    ):
        super().__init__()
        self.imsize = imsize
        self.inst_norm_idt = inst_norm_idt

        self.e_pri = ZEncoder(
            input_imsize=imsize,
            smallest_imsize=4,
            nc_in=4,
            nc_base=nc_base,
            n_layers_per_res=[2, 4, 8, 4, 2],
        )

        self.e_mod = StyleEncoder(
            imsize=imsize,
            smallest_imsize=4,
            nc_base=nc_base,
            to_z=False,
        )

        self.f = FcLayers(4, 1.)

    def forward(self, img, gender):
        assert len(gender.shape) == 1
        gender = torch.reshape(gender, (-1, 1, 1, 1))
        gender = gender.repeat(1, 1, self.imsize, self.imsize)

        z = self.e_pri(torch.cat([img, gender], dim=1))
        idt = self.e_mod(img, z)
        delta = self.f(z)

        if self.inst_norm_idt:
            idt = F.instance_norm(idt)

        return idt, z, delta
