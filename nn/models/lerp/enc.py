#!/usr/bin/env python3
import torch
import torch.nn as nn
from ai_old.util.params import init_params
import external.sg2.misc as misc
from ai_old.nn.models.encode.fm2l import FcFeatMapToLatent
from ai_old.nn.blocks.conv import ConvBlock
from ai_old.nn.blocks.res import ResBlocks
from external.sg2.unit import SynthesisLayer, Conv2dLayer
import numpy as np
from copy import deepcopy
from ai_old.nn.blocks.onnx import OnnxConv2dLayer, OnnxEncSynthesisLayer
from ai_old.nn.blocks.mod import DoubleExcitationBlock


def dynamic_to_learned_const(enc_lerper, base_enc, guide_enc):
    assert isinstance(enc_lerper, EncLerpV1)
    misc.assert_shape(base_enc, [None, 512, 4, 4])
    misc.assert_shape(guide_enc, [None, 512, 4, 4])

    combo = torch.cat([base_enc, guide_enc], dim=1)
    identity = enc_lerper.identity_encoder(combo)
    base_w = enc_lerper.w_encoder(combo)
    delta = enc_lerper.delta_encoder(combo)

    enc_lerp_lc = EncLerpLearnedConsts(identity, base_w, delta)
    enc_lerp_g = GenOnlyEncLerpV1(enc_lerper.enc_generator)

    return enc_lerp_lc, enc_lerp_g


class OnnxStyleEncGeneratorBlock(nn.Module):
    def __init__(self, actv='lrelu'):
        super().__init__()

        self.conv0 = OnnxEncSynthesisLayer(
            512,
            512,
            w_dim=512,
            activation=actv,
        )

        self.conv1 = OnnxEncSynthesisLayer(
            512,
            512,
            w_dim=512,
            activation=actv,
        )

        self.skip = OnnxConv2dLayer(
            512,
            512,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x, w):
        y = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv0(x, w)
        x = self.conv1(x, w, gain=np.sqrt(0.5))
        x = y.add_(x)
        return x


class StyleEncGeneratorBlock(nn.Module):
    def __init__(self, actv='lrelu'):
        super().__init__()

        self.conv0 = SynthesisLayer(
            512,
            512,
            w_dim=512,
            resolution=4,
            up=1,
            conv_clamp=None,
            activation=actv,
        )

        self.conv1 = SynthesisLayer(
            512,
            512,
            w_dim=512,
            resolution=4,
            up=1,
            conv_clamp=None,
            activation=actv,
        )

        self.skip = Conv2dLayer(
            512,
            512,
            kernel_size=1,
            bias=False,
            up=1,
        )

    def forward(self, x, w):
        y = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv0(
            x,
            w,
            fused_modconv=None,
            noise_mode='none',
        )
        x = self.conv1(
            x,
            w,
            fused_modconv=None,
            gain=np.sqrt(0.5),
            noise_mode='none',
        )
        x = y.add_(x)
        return x


# class ExcitationEncGeneratorBlock(nn.Module):
#     def __init__(self, norm='batch', actv='lrelu', onnx=False):
#         super().__init__()
#
#         self.conv0 = ExcitationBlockV2(
#             512,
#             512,
#             512,
#             norm=norm,
#             actv=actv,
#             onnx=onnx,
#         )
#
#         self.conv1 = ExcitationBlockV2(
#             512,
#             512,
#             512,
#             norm=norm,
#             actv=actv,
#             onnx=onnx,
#         )
#
#         if onnx:
#             skip_cls = OnnxConv2dLayer
#         else:
#             skip_cls = Conv2dLayer
#         self.skip = skip_cls(
#             512,
#             512,
#             kernel_size=1,
#             bias=False,
#             up=1,
#         )
#
#     def forward(self, x, w):
#         y = self.skip(x, gain=np.sqrt(0.5))
#         x = self.conv0(
#             x,
#             w,
#         )
#         x = self.conv1(
#             x,
#             w,
#             gain=np.sqrt(0.5),
#         )
#         x = y.add_(x)
#         return x


class EncGenerator(nn.Module):
    def __init__(self,
        n_blocks=4,
        final_conv_k=None,
        type='style',
        norm='batch',
        actv='lrelu',
        onnx=False,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        for i in range(n_blocks):
            if type == 'style':
                if onnx:
                    block = OnnxStyleEncGeneratorBlock(actv=actv)
                else:
                    block = StyleEncGeneratorBlock(actv=actv)
            elif type == 'excitation':
                block = DoubleExcitationBlock(
                    512,
                    512,
                    512,
                    k_shortcut=1,
                    norm=norm,
                    actv=actv,
                )
            else:
                raise Exception(type)
            setattr(self, f'b{i}', block)

        if final_conv_k is not None:
            self.final = nn.Conv2d(
                512,
                512,
                kernel_size=final_conv_k,
                padding=(final_conv_k - 1) // 2,
            )
        else:
            self.final = None

    def forward(self, x, w):
        for i in range(self.n_blocks):
            x = getattr(self, f'b{i}')(x, w)
        if self.final is not None:
            x = self.final(x)
        return x


class EncLerpV0(nn.Module):
    def __init__(self,
        cfg,
        n_id_res_blocks=2,
        n_w_res_blocks=1,
        n_delta_res_blocks=1,
        n_gen_blocks=4,
        final_conv_k=None,
        g_type='style',
        g_norm='batch', # if type != 'style'
        g_actv='lrelu',
        e_actv='mish',
        onnx=False,
    ):
        super().__init__()

        self.identity_encoder = nn.Sequential(
            ConvBlock(1024, 512, actv=e_actv),
            ResBlocks(512, n_id_res_blocks),
        )

        self.w_encoder = nn.Sequential(
            ConvBlock(1024, 512, actv=e_actv),
            ResBlocks(512, n_w_res_blocks, actv=e_actv),
            FcFeatMapToLatent(512),
        )

        self.delta_encoder = nn.Sequential(
            ConvBlock(1024, 512, actv=e_actv),
            ResBlocks(512, n_delta_res_blocks, actv=e_actv),
            FcFeatMapToLatent(512),
        )

        self.enc_generator = EncGenerator(
            n_blocks=n_gen_blocks,
            final_conv_k=final_conv_k,
            type=g_type,
            norm=g_norm,
            actv=g_actv,
            onnx=onnx,
        )

        self.apply(init_params())

    def forward(self, base_enc, guide_enc, mag):
        misc.assert_shape(base_enc, [None, 512, 4, 4])
        misc.assert_shape(guide_enc, [None, 512, 4, 4])
        combo = torch.cat([base_enc, guide_enc], dim=1)
        identity = self.identity_encoder(combo)
        base_w = self.w_encoder(combo)
        delta = self.delta_encoder(combo)
        w = base_w + delta * mag
        return self.enc_generator(identity, w)

    def prep_for_train_phase(self):
        self.requires_grad_(True)


class EncLerpV1(EncLerpV0):
    def forward(self, base_enc, guide_enc, mag):
        bs = base_enc.shape[0]
        misc.assert_shape(base_enc, [bs, 512, 4, 4])
        misc.assert_shape(guide_enc, [bs, 512, 4, 4])
        # bs2 = mag.shape[0]
        # misc.assert_shape(mag, [bs2])
        misc.assert_shape(mag, [bs])

        combo = torch.cat([base_enc, guide_enc], dim=1)
        identity = self.identity_encoder(combo)
        base_w = self.w_encoder(combo)
        delta = self.delta_encoder(combo)

        # if bs1 != bs2:
        #     assert bs1 == 1
        #     identity = identity.repeat(bs2, 1, 1, 1)
        #     base_w = base_w.repeat(bs2, 1)
        #     delta = delta.repeat(bs2, 1)

        w = base_w + delta * torch.reshape(mag, (-1, 1))
        enc_delta = self.enc_generator(identity, w)
        return base_enc + enc_delta * torch.reshape(mag, (-1, 1, 1, 1))


class GenOnlyEncLerpV1(nn.Module):
    def __init__(self, enc_generator):
        super().__init__()
        self.enc_generator = deepcopy(enc_generator)

    def forward(self, base_enc, identity, base_w, delta, mag):
        bs1 = base_enc.shape[0]
        bs2 = mag.shape[0]
        if bs1 != bs2:
            assert bs1 == 1
            identity = identity.repeat(bs2, 1, 1, 1)
            base_w = base_w.repeat(bs2, 1)
            delta = delta.repeat(bs2, 1)

        w = base_w + delta * torch.reshape(mag, (-1, 1))
        enc_delta = self.enc_generator(identity, w)
        return base_enc + enc_delta * torch.reshape(mag, (-1, 1, 1, 1))


class EncLerpLearnedConsts(nn.Module):
    def __init__(self, identity, base_w, delta):
        super().__init__()
        self.identity = nn.Parameter(identity.clone().detach())
        self.base_w = nn.Parameter(base_w.clone().detach())
        self.delta = nn.Parameter(delta.clone().detach())

    def forward(self):
        return self.identity, self.base_w, self.delta
