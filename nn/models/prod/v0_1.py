#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_old.util.factory import build_model_from_exp, build_model
from ai_old.util.pretrained import build_pretrained_e4e, build_pretrained_sg2
import numpy as np
from ai_old.trainer.pti import PtiTrainer
from ai_old.util.etc import resize_imgs, pil_to_tensor, AttrDict, \
    normalized_tensor_to_pil_img
from ai_old.util.inverse import solo_aligned_to_fam_final
from ai_old.nn.models.seg.seg import Segmenter, FHBC_GROUPS
from ai_old.util.outer import resize_and_pad_inner_img, fhbc_seg_to_facehair, \
    dilate_mask, get_inner_mask, get_outer_boundary_mask, get_dilate_kernel, \
    outer_align
from ai_old.nn.models.inpaint.aot import AotInpainter
import external.sg2.misc as misc
from copy import deepcopy
from ai_old.finetune.enc_lerp import finetune_enc_lerp
from ai_old.finetune.ae import finetune_ae


W_LERP_EXP = 'lerp/5/5'
AE_EXP = 'rec/25/8'
AE_IS_ONNX_READY = False
ENC_LERP_EXP = 'enc-lerp/1/2'
ENC_LERP_IS_ONNX_READY = True
OUTER_SEG_PRED_EXP = 'outer-seg/0/1'

MAX_MAG = 1.5
FT_N_MAGS = 8


class ProdClientModel(nn.Module):
    def __init__(self, enc_generator, img_generator):
        super().__init__()
        self.enc_generator = deepcopy(enc_generator)
        self.img_generator = deepcopy(img_generator)

    def forward(self, base_enc, identity, base_latent, delta, mag):
        misc.assert_shape(base_enc, [1, 512, 4, 4])
        misc.assert_shape(identity, [1, 512, 4, 4])
        misc.assert_shape(base_latent, [1, 512])
        misc.assert_shape(delta, [1, 512])
        misc.assert_shape(mag, [1])

        # enc
        latent = base_latent + delta * torch.reshape(mag, (-1, 1))
        enc_delta = self.enc_generator(identity, latent)
        enc = base_enc + enc_delta * torch.reshape(mag, (-1, 1, 1, 1))

        # img
        return self.img_generator(enc, noise_mode='const')


def get_prod_client():
    enc_lerper, cfg = build_model_from_exp(ENC_LERP_EXP, 'model')
    enc_lerper = enc_lerper.eval().to('cuda')
    if not ENC_LERP_IS_ONNX_READY:
        setattr(cfg.model, 'onnx', True)
        onnx_enc_lerper = build_model(cfg, cfg.model).eval().to('cuda')
        onnx_enc_lerper.load_state_dict(enc_lerper.state_dict(), strict=False)
        enc_lerper = onnx_enc_lerper

    ae, cfg = build_model_from_exp(AE_EXP, 'G_ema')
    ae = ae.eval().to('cuda')
    if not AE_IS_ONNX_READY:
        setattr(cfg.model.G, 'onnx', True)
        onnx_ae = build_model(cfg, cfg.model.G).eval().to('cuda')
        onnx_ae.load_state_dict(ae.state_dict(), strict=False)
        ae = onnx_ae

    return ProdClientModel(
        enc_lerper.enc_generator,
        ae.g,
    ).eval().to('cuda')


class ProdServerModel(nn.Module):
    def __init__(self, f, g, w, is_mtf):
        super().__init__()
        self.f = f
        self.g = g
        self.register_buffer('w', w.clone().detach())
        self.register_buffer('is_mtf', is_mtf.clone().detach())

    def forward(self, mag):
        return self.g(
            self.f(self.w, self.is_mtf, magnitude=mag),
            noise_mode='const',
        )


class Prod:
    def __init__(self):
        self.E = build_pretrained_e4e().to('cuda').eval()

        self.F_w = build_model_from_exp(
            W_LERP_EXP,
            'G',
            return_cfg=False,
        ).f.eval().to('cuda')

        self.AE = build_model_from_exp(
            AE_EXP,
            'G_ema',
            return_cfg=False,
        ).eval().to('cuda')

        self.F_enc = build_model_from_exp(
            ENC_LERP_EXP,
            'model',
            return_cfg=False,
        ).eval().to('cuda')

        self.segmenter = Segmenter().to('cuda')
        self.segmenter.eval()

        self.outer_seg_predictor = build_model_from_exp(
            OUTER_SEG_PRED_EXP,
            'model',
            return_cfg=False,
            migrate_to_pt=False,
        ).eval().to('cuda')
        assert not self.outer_seg_predictor.pred_from_seg_only

        inpainter = AotInpainter(AttrDict({'dataset': AttrDict(
            {'imsize': 512})}))
        path = '/home/asiu/data/models/aot/G.pt'
        inpainter.load_state_dict(torch.load(path, map_location='cuda'))
        self.inpainter = inpainter.to('cuda').eval()

        self.pti_trainer = PtiTrainer('cuda')

        self.mags = torch.tensor(
            np.linspace(0., MAX_MAG, num=FT_N_MAGS),
            device='cuda',
        ).to(torch.float32)

    def run(self, aligned, aligned_256, is_mtf, debug=False):
        assert aligned.shape == (1, 3, 1024, 1024)
        assert aligned_256.shape == (1, 3, 256, 256)
        assert is_mtf.shape == (1,)

        # encode
        with torch.no_grad():
            w = self.E(aligned_256)

        # finetune G
        G = self.pti_trainer.train(aligned, w).synthesis

        # calc target imgs
        with torch.no_grad():
            target_img = resize_imgs(G(self.F_w(
                w.repeat(FT_N_MAGS, 1, 1),
                is_mtf.repeat(FT_N_MAGS, 1),
                magnitude=self.mags,
            )), 256)

        # finetune AE
        AE_lc = finetune_ae(
            self.AE,
            target_img,
        )

        # calc encs
        with torch.no_grad():
            target_enc = AE_lc()
            base_enc = target_enc[0, :, :, :].unsqueeze(0)
            guide_enc = target_enc[FT_N_MAGS - 1, :, :, :].unsqueeze(0)

        # finetune enc lerp
        F_enc_lc, F_enc_g = finetune_enc_lerp(
            self.F_enc,
            base_enc,
            guide_enc,
            target_enc,
            self.mags,
        )

        # wrap client inputs
        identity, base_latent, delta = F_enc_lc()
        misc.assert_shape(base_enc, [1, 512, 4, 4])
        misc.assert_shape(identity, [1, 512, 4, 4])
        misc.assert_shape(base_latent, [1, 512])
        misc.assert_shape(delta, [1, 512])
        client_inputs = {
            'base_enc': base_enc,
            'identity': identity,
            'base_latent': base_latent,
            'delta': delta,
        }

        # wrap server model pieces
        server_model = ProdServerModel(self.F_w, G, w, is_mtf).eval()

        return client_inputs, server_model

    def finalize(self, full_img, inner_quad, state_dict, magnitude):
        outer_aligned, outer_quad = outer_align(full_img, inner_quad)
        outer_aligned = pil_to_tensor(outer_aligned)

        server_model = self._build_empty_server_model()
        server_model.load_state_dict(state_dict)
        server_model.requires_grad_(False)
        server_model.eval()

        swap_aligned = server_model(magnitude)

        inner_imsize = int(512 / 1.5)
        outer_imsize = 512

        # segment
        swap_seg = torch.argmax(self.segmenter(
            swap_aligned,
            groups=FHBC_GROUPS,
            output_imsize=128,
        )[0], dim=0)
        outer_seg = torch.argmax(self.segmenter(
            outer_aligned,
            groups=FHBC_GROUPS,
            output_imsize=outer_imsize,
        )[0], dim=0)

        # predict swap outer seg from swap inner seg
        swap_outer_seg = self.outer_seg_predictor(
            resize_and_pad_inner_img(
                swap_seg.unsqueeze(0),
                self.outer_seg_predictor.inner_mask,
                128,
                192,
                is_seg=True,
            ),
            resize_and_pad_inner_img(
                swap_aligned,
                self.outer_seg_predictor.inner_mask,
                128,
                192,
            ),
        )
        swap_outer_seg = resize_imgs(swap_outer_seg, outer_imsize)
        swap_outer_seg = torch.argmax(swap_outer_seg[0], dim=0)

        # calc masks
        facehair = fhbc_seg_to_facehair(outer_seg)
        swap_facehair = fhbc_seg_to_facehair(swap_outer_seg)
        inner_mask = get_inner_mask(inner_imsize, outer_imsize)
        inner_gan_mask = swap_facehair * inner_mask
        inv_inner_gan_mask = 1. - inner_gan_mask
        dilate_kernel = get_dilate_kernel(outer_imsize)
        dilated_facehair = dilate_mask(facehair, dilate_kernel)
        dilated_swap_facehair = dilate_mask(swap_facehair, dilate_kernel)
        outer_boundary_mask = get_outer_boundary_mask(outer_imsize).to('cuda')
        inv_outer_boundary_mask = 1. - outer_boundary_mask
        dilated_facehair_union = torch.clamp(
            dilated_facehair + dilated_swap_facehair, min=0., max=1.)
        inpaint_mask = dilated_facehair_union * inv_inner_gan_mask * \
            inv_outer_boundary_mask
        gt_mask = torch.ones_like(inpaint_mask) * (1. - inpaint_mask) * \
            inv_inner_gan_mask

        # merge and inpaint
        padded_swap = resize_and_pad_inner_img(
            swap_aligned, inner_mask, inner_imsize, outer_imsize)
        outer_aligned = resize_imgs(outer_aligned, outer_imsize)
        merged = outer_aligned * gt_mask + padded_swap * inner_gan_mask
        inpainted = self.inpainter(merged, inpaint_mask.unsqueeze(0))[0]
        inpainted = normalized_tensor_to_pil_img(inpainted)

        # reinsert into full img
        final_full = solo_aligned_to_fam_final(
            inpainted,
            outer_quad,
            full_img,
        )
        return final_full

    def _build_empty_server_model(self):
        return ProdServerModel(
            self.F_w,
            build_pretrained_sg2(load_weights=False).synthesis,
            torch.zeros((1, 18, 512), device='cuda'),
            torch.zeros((1,), device='cuda'),
        ).cuda()
