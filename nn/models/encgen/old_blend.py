#!/usr/bin/env python3
import torch
import torch.nn as nn
from external.sg2 import persistence
from external.sg2.unit import SynthesisBlock, FullyConnectedLayer, modulated_conv2d
from ai_old.nn.models.encode.squeeze import ZEncoder
from ai_old.util import config
import copy
from ai_old.util.params import init_params
from ai_old.util.factory import build_model_from_exp
from ai_old.util.config import get_default
import numpy as np
from ai_old.nn.blocks.blend import BlendBlock
from ai_old.nn.blocks.res import ResDownConvBlock
import external.sg2.misc as misc
from external.op import upfirdn2d
from external.op import bias_act


@persistence.persistent_class
class BlendSynthesisNetwork(nn.Module):
    def __init__(self,
        w_dim,
        img_resolution,
        img_channels,
        channel_base=32768,
        channel_max=512,
        num_fp16_res=0,
        k_blend=3,
        **block_kwargs,
    ):
        assert img_resolution >= 4 and \
            img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [
            2 ** i for i in range(2, self.img_resolution_log2 + 1)
        ]
        channels_dict = {
            res: min(channel_base // res, channel_max) \
            for res in self.block_resolutions
        }
        fp16_resolution = max(
            2 ** (self.img_resolution_log2 + 1 - num_fp16_res),
            8,
        )

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=res,
                img_channels=img_channels,
                is_last=is_last,
                use_fp16=use_fp16,
                **block_kwargs,
            )
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)
            if not is_last:
                setattr(self, f'blend{res}', BlendBlock(
                    out_channels,
                    k=k_blend,
                    use_fp16=use_fp16,
                ))

    def forward(self, ibg_encs, ws, **block_kwargs):
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
            if i < len(self.block_resolutions) - 1:
                x = getattr(self, f'blend{res}')(x, ibg_encs[i])
        return img


@persistence.persistent_class
class StyleDownBlock(torch.nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        w_dim,
        kernel_size=3,
        activation='lrelu',
        resample_filter=[1,3,3,1],
        conv_clamp=None,
    ):
        super().__init__()
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer(
            'resample_filter',
            upfirdn2d.setup_filter(resample_filter),
        )
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
        ]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, fused_modconv=True, gain=1):
        styles = self.affine(w)

        x = modulated_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
            noise=None,
            up=1,
            down=2,
            padding=self.padding,
            resample_filter=self.resample_filter,
            flip_weight=False,
            fused_modconv=fused_modconv,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain \
            if self.conv_clamp is not None else None
        x = bias_act.bias_act(
            x,
            self.bias.to(x.dtype),
            act=self.activation,
            gain=act_gain,
            clamp=act_clamp,
        )
        return x


@persistence.persistent_class
class IbgStyleEncoder(nn.Module):
    def __init__(self,
        img_resolution,
        img_channels,
        channel_base=32768,
        channel_max=512,
        w_dim=512,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [
            2 ** i for i in range(2, self.img_resolution_log2 + 1)
        ]
        channels_dict = {
            res: min(channel_base // res, channel_max) \
            for res in self.block_resolutions
        }

        for res in self.block_resolutions:
            if res > 4:
                out_channels = channels_dict[res // 2]
                in_channels = 3 if res == self.img_resolution else \
                    channels_dict[res]
                block = StyleDownBlock(
                    in_channels,
                    out_channels,
                    w_dim=w_dim,
                )
                setattr(self, f'b{res}', block)

    def forward(self, ibg, w):
        encs = []
        x = ibg
        for res in self.block_resolutions[::-1]:
            if res > 4:
                block = getattr(self, f'b{res}')
                x = block(x, w)
                encs.append(x)
        return encs[::-1]


@persistence.persistent_class
class IbgEncoder(nn.Module):
    def __init__(self,
        img_resolution,
        img_channels,
        channel_base=32768,
        channel_max=512,
        norm='instance',
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [
            2 ** i for i in range(2, self.img_resolution_log2 + 1)
        ]
        channels_dict = {
            res: min(channel_base // res, channel_max) \
            for res in self.block_resolutions
        }

        for res in self.block_resolutions:
            if res > 4:
                out_channels = channels_dict[res // 2]
                in_channels = 3 if res == self.img_resolution else \
                    channels_dict[res]
                block = ResDownConvBlock(
                    in_channels,
                    out_channels,
                    norm=norm,
                )
                setattr(self, f'b{res}', block)

    def forward(self, ibg):
        encs = []
        x = ibg
        for res in self.block_resolutions[::-1]:
            if res > 4:
                block = getattr(self, f'b{res}')
                x = block(x)
                encs.append(x)
        return encs[::-1]


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
        e_ibg_type='style'
    ):
        super().__init__()
        self.e_ibg_type = e_ibg_type

        # load the original generator and its config
        og_G, og_cfg = build_model_from_exp(g_exp, 'G')

        # hparams
        self.z_dims = get_default(og_cfg.model.G, 'z_dims', 512)
        self.num_ws = og_G.f.num_ws
        self.imsize = get_default(og_cfg.model.G, 'imsize', 128)
        nc_in = get_default(og_cfg.model.G, 'nc_in', 3)
        nc_base = get_default(og_cfg.model.G, 'nc_base', 32)
        nc_max = get_default(og_cfg.model.G, 'nc_max', 512)
        num_fp16_res = get_default(og_cfg.model.G, 'num_fp16_res', 4)

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
            k_blend=g_k_blend,
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

        # build ibg encoder
        if self.e_ibg_type == 'res':
            self.e_ibg = IbgEncoder(
                img_resolution=self.imsize,
                img_channels=nc_in,
                channel_base=nc_base * self.imsize,
                channel_max=nc_max,
                norm='instance',
            )
        elif self.e_ibg_type == 'style':
            self.e_ibg = IbgStyleEncoder(
                img_resolution=self.imsize,
                img_channels=nc_in,
                channel_base=nc_base * self.imsize,
                channel_max=nc_max,
                w_dim=self.z_dims,
            )
        self.e_ibg.apply(init_params())

    def forward(self, fg, ibg):
        w = self.e(fg)
        ws = w.unsqueeze(1).repeat([1, self.num_ws, 1])
        if self.e_ibg_type == 'style':
            ibg_encs = self.e_ibg(ibg, w)
        else:
            ibg_encs = self.e_ibg(ibg)
        output = self.g(ibg_encs, ws)
        return output

    def prep_for_train_phase(self):
        self.e.requires_grad_(True)
        self.e_ibg.requires_grad_(True)
