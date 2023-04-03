#!/usr/bin/env python3
import torch.nn as nn
from external.sg2 import persistence
from ai_old.util.params import init_params
from ai_old.nn.models.decode.noise import NoiseResDecoder
from ai_old.nn.models.encode.squeeze import Encoder as SqueezeEncoder
from ai_old.nn.models.encode.simple import SimpleEncoder
from copy import deepcopy
import external.sg2.misc as misc


def dynamic_to_learned_const(ae, img):
    assert isinstance(ae, NoiseAutoencoder)
    misc.assert_shape(img, [None, 3, ae.imsize, ae.imsize])
    enc = ae.e(img)
    ae_lc = EncLearnedConst(enc)
    return ae_lc


class EncLearnedConst(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = nn.Parameter(enc.clone().detach())

    def forward(self):
        return self.enc


@persistence.persistent_class
class NoiseAutoencoder(nn.Module):
    def __init__(self,
        cfg,
        e_type='squeeze',
        smallest_imsize=4,
        nc_in=3,
        nc_base=64,
        nc_max=512,
        n_layers_per_res=[2, 2, 4, 8, 4, 2],
        norm='batch',
        actv='mish',
        conv_clamp=None,
        dropout_after_squeeze_layers=False,
        onnx=False,
        simple=False,
    ):
        super().__init__()
        self.intermediate = 'enc'
        self.imsize = cfg.dataset.imsize

        if e_type == 'squeeze':
            self.e = SqueezeEncoder(
                input_imsize=self.imsize,
                smallest_imsize=smallest_imsize,
                nc_in=nc_in,
                nc_base=nc_base,
                nc_max=nc_max,
                n_layers_per_res=n_layers_per_res,
                norm=norm,
                actv=actv,
                conv_clamp=conv_clamp,
                dropout_after_squeeze_layers=dropout_after_squeeze_layers,
            )
            self.e.apply(init_params())
        elif e_type == 'simple':
            self.e = SimpleEncoder(
                input_imsize=self.imsize,
                smallest_imsize=smallest_imsize,
                nc_in=nc_in,
                nc_base=nc_base,
                nc_max=nc_max,
                norm=norm,
                actv=actv,
                conv_clamp=conv_clamp,
            )
            self.e.apply(init_params())
        else:
            raise Exception(e_type)

        self.g = NoiseResDecoder(
            imsize=self.imsize,
            smallest_imsize=smallest_imsize,
            nc_in=nc_in,
            nc_base=nc_base,
            nc_max=nc_max,
            norm=norm,
            actv=actv,
            conv_clamp=conv_clamp,
            onnx=onnx,
            simple=simple,
        )
        self.g.apply(init_params())


    def forward(self, x, noise_mode='random'):
        encoding = self.e(x)
        return self.g(encoding, noise_mode=noise_mode)

    def prep_for_train_phase(self):
        self.requires_grad_(True)
