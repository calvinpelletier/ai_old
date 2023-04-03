#!/usr/bin/env python3
import torch
import torch.nn as nn
from external.sg2 import persistence
from external.sg2.unit import SynthesisNetwork
from ai_old.nn.models.encode.squeeze import ZEncoder
from ai_old.util import config
import copy
from ai_old.util.params import init_params
from ai_old.util.factory import build_model_from_exp
from ai_old.nn.blocks.blend import MetaBlendBlock
from ai_old.nn.blocks.res import ResDownConvBlock, ResUpConvBlock
from ai_old.nn.blocks.conv import ConvBlock, ConvToImg
import external.sg2.misc as misc


@persistence.persistent_class
class BlendSynthesisNetwork(SynthesisNetwork):
    def forward(self, ws, **block_kwargs):
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

        x = img = None
        for i, (res, cur_ws) in enumerate(zip(self.block_resolutions, block_ws)):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img, x.to(torch.float32)


@persistence.persistent_class
class IbgBlender(nn.Module):
    def __init__(self,
        nc_base=32,
        n_down_up=2,
        k_blend=3,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()
        nc_in = 3

        # fg encoder
        fg_enc_blocks = [ConvBlock(
            nc_in,
            nc_base,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )]
        for i in range(n_down_up):
            fg_enc_blocks.append(ResDownConvBlock(
                nc_base * 2 ** i,
                nc_base * 2 ** (i + 1),
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            ))
        self.fg_encoder = nn.Sequential(*fg_enc_blocks)

        # other blocks
        blend_blocks = [MetaBlendBlock(
            nc_base,
            k=k_blend,
        )]
        dec_blocks = [ConvToImg(nc_base)]
        ibg_enc_blocks = [ConvBlock(
            nc_in,
            nc_base,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )]
        meta_enc_blocks = [ConvBlock(
            nc_base,
            nc_base,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )]
        for i in range(n_down_up):
            blend_blocks.append(MetaBlendBlock(
                nc_base * 2 ** (i + 1),
                k=k_blend,
            ))
            dec_blocks.append(ResUpConvBlock(
                nc_base * 2 ** (i + 1),
                nc_base * 2 ** i,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            ))
            ibg_enc_blocks.append(ResDownConvBlock(
                nc_base * 2 ** i,
                nc_base * 2 ** (i + 1),
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            ))
            meta_enc_blocks.append(ResDownConvBlock(
                nc_base * 2 ** i,
                nc_base * 2 ** (i + 1),
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            ))
        self.blend_blocks = nn.ModuleList(blend_blocks[::-1])
        self.dec_blocks = nn.ModuleList(dec_blocks[::-1])
        self.ibg_enc_blocks = nn.ModuleList(ibg_enc_blocks)
        self.meta_enc_blocks = nn.ModuleList(meta_enc_blocks)

    def forward(self, fg, ibg, meta):
        x_ibg = ibg
        x_meta = meta
        ibg_encs = []
        meta_encs = []
        for ibg_enc_block, meta_enc_block in zip(
            self.ibg_enc_blocks,
            self.meta_enc_blocks,
        ):
            x_ibg = ibg_enc_block(x_ibg)
            x_meta = meta_enc_block(x_meta)
            ibg_encs.append(x_ibg)
            meta_encs.append(x_meta)
        ibg_encs = ibg_encs[::-1]
        meta_encs = meta_encs[::-1]

        x = self.fg_encoder(fg)
        for i, (blend_block, dec_block) in enumerate(zip(
            self.blend_blocks,
            self.dec_blocks,
        )):
             x = blend_block(x, ibg_encs[i], meta_encs[i])
             x = dec_block(x)
        return x


@persistence.persistent_class
class GeneratorInitializedBlendAutoencoder(nn.Module):
    def __init__(self,
        cfg,
        g_exp='facegen/8/1',
        g_k_blend=3,
        e_type='slow_squeeze_excite',
        e_nc_base=32,
        e_n_layers_per_res=[2, 4, 8, 4, 2],
        e_norm='batch',
        e_weight_norm=False,
        e_actv='mish',
        e_ibg_norm='instance',
        b_k_blend=3,
        b_n_down_up=1,
    ):
        super().__init__()

        # load the original generator and its config
        og_G, og_cfg = build_model_from_exp(g_exp, 'G')

        # hparams
        self.z_dims = config.get_default(og_cfg.model.G, 'z_dims', 512)
        self.num_ws = og_G.f.num_ws
        self.imsize = config.get_default(og_cfg.model.G, 'imsize', 128)
        nc_in = config.get_default(og_cfg.model.G, 'nc_in', 3)
        nc_base = config.get_default(og_cfg.model.G, 'nc_base', 32)
        nc_max = config.get_default(og_cfg.model.G, 'nc_max', 512)
        num_fp16_res = config.get_default(og_cfg.model.G, 'num_fp16_res', 4)

        # create blend synth network
        self.g = BlendSynthesisNetwork(
            w_dim=self.z_dims,
            img_resolution=self.imsize,
            img_channels=nc_in,
            channel_base=nc_base * self.imsize,
            channel_max=nc_max,
            num_fp16_res=num_fp16_res,
            conv_clamp=256,
            fp16_channels_last=False,
        )
        self.g.requires_grad_(False)

        # transfer params from the original generator
        for res in self.g.block_resolutions:
            setattr(
                self.g,
                f'b{res}',
                copy.deepcopy(getattr(og_G.g, f'b{res}')),
            )

        del og_G

        # build encoder
        assert e_type == 'slow_squeeze_excite'
        self.e = ZEncoder(
            input_imsize=self.imsize,
            smallest_imsize=4,
            z_dims=self.z_dims,
            nc_in=3,
            nc_base=e_nc_base,
            n_layers_per_res=e_n_layers_per_res,
            norm=e_norm,
            weight_norm=e_weight_norm,
            actv=e_actv,
        )
        self.e.apply(init_params())

        # build blender
        self.b = IbgBlender(
            nc_base=nc_base,
            k_blend=b_k_blend,
            n_down_up=b_n_down_up,
        )
        self.b.apply(init_params())

    def forward(self, fg, ibg):
        w = self.e(fg)
        ws = w.unsqueeze(1).repeat([1, self.num_ws, 1])
        fg_out, meta_out = self.g(ws)
        full_out = self.b(fg_out, ibg, meta_out)
        return full_out

    def prep_for_train_phase(self):
        self.e.requires_grad_(True)
        self.b.requires_grad_(True)
