#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_old.util.factory import build_model_from_exp
from ai_old.util.pretrained import build_pretrained_e4e, build_pretrained_sg2
import numpy as np
import cv2
from lpips import LPIPS
from random import random
from ai_old.nn.models.lerp.onnx import OnnxLevelsDynamicLerper
from ai_old.trainer.pti import PtiTrainer
from tqdm import tqdm
from ai_old.util.etc import resize_imgs, pil_to_tensor, AttrDict, \
    normalized_tensor_to_pil_img
from ai_old.util.inverse import get_outer_quad, solo_aligned_to_fam_final
from ai_old.util.face import custom_align_face
from ai_old.nn.models.seg.seg import Segmenter, FHBC_GROUPS
from ai_old.util.outer import resize_and_pad_inner_img, fhbc_seg_to_facehair, \
    dilate_mask, get_inner_mask, get_outer_boundary_mask, get_dilate_kernel
from ai_old.nn.models.inpaint.aot import AotInpainter


ALIGNED_IMSIZE = 256
MAX_MAG = 1.5

STUDENT_EXP = 'distill/1/1'
STUDENT_IMSIZE = 256

STUDENT_FT_LR = 3e-4
STUDENT_FT_MAX_ITER = 400
STUDENT_FT_EARLY_STOP = 0.03


def outer_align(full_img, inner_quad):
    outer_quad = get_outer_quad(inner_quad, full=full_img)
    outer_imsize = 1024 + 512
    outer_aligned = custom_align_face(full_img, outer_quad, outer_imsize)
    return outer_aligned, outer_quad


class _Teacher(nn.Module):
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


class _Student(nn.Module):
    def __init__(self, f, g, w, is_mtf):
        super().__init__()
        self.f = f
        self.g = g
        self.register_buffer('w', w.clone().detach())
        self.register_buffer('is_mtf', is_mtf.clone().detach())

    def forward(self, mag):
        return self.g(self.f(self.w, self.is_mtf, magnitude=mag))


class ProdV1:
    def __init__(self, exp):
        # TODO: exp

        self.aligned_imsize = ALIGNED_IMSIZE

        self.e = build_pretrained_e4e().to('cuda').eval()

        self.pti_trainer = PtiTrainer('cuda')

        f_model, f_cfg = build_model_from_exp('lerp/5/5', 'G')
        self.f = OnnxLevelsDynamicLerper(
            levels=f_cfg.model.levels,
            final_activation=f_cfg.model.final_activation,
            mult=f_cfg.model.mult,
            lr_mul=f_cfg.model.lr_mul,
            pixel_norm=f_cfg.model.pixel_norm,
        )
        self.f.load_state_dict(f_model.f.state_dict())
        self.f = self.f.to('cuda').eval()

        self.segmenter = Segmenter().to('cuda')
        self.segmenter.eval()

        self.outer_seg_predictor = build_model_from_exp(
            'outer-seg/0/1',
            'model',
            return_cfg=False,
        ).to('cuda')
        self.outer_seg_predictor.eval()
        assert not self.outer_seg_predictor.pred_from_seg_only

        inpainter = AotInpainter(AttrDict({'dataset': AttrDict(
            {'imsize': 512})}))
        path = '/home/asiu/data/models/aot/G.pt'
        inpainter.load_state_dict(torch.load(path, map_location='cuda'))
        self.inpainter = inpainter.to('cuda').eval()


    def run(self, aligned, aligned_256, is_mtf, debug=False):
        assert aligned.shape == (1, 3, 1024, 1024)
        assert aligned_256.shape == (1, 3, 256, 256)
        assert is_mtf.shape == (1,)

        with torch.no_grad():
            # encode
            w = self.e(aligned_256)

        # finetune G
        teacher_G = self.pti_trainer.train(aligned, w).synthesis

        # finetune student
        student_G = self._finetune_student(teacher_G, w, is_mtf)

        # wrap editor, generator, and const params into one class
        teacher = _Teacher(self.f, teacher_G, w, is_mtf).eval()
        student = _Student(self.f, student_G, w, is_mtf).eval()

        return teacher, student


    def finalize(self, full_img, inner_quad, state_dict, magnitude):
        outer_aligned, outer_quad = outer_align(full_img, inner_quad)
        outer_aligned = pil_to_tensor(outer_aligned)

        teacher = self._build_empty_teacher()
        teacher.load_state_dict(state_dict)
        teacher.requires_grad_(False)
        teacher.eval()

        swap_aligned = teacher(magnitude)

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


    def _finetune_student(self, teacher, w, is_mtf):
        print('finetuning student...')
        w = w.to('cuda')
        is_mtf = is_mtf.to('cuda')

        teacher.requires_grad_(False)
        teacher.eval()

        self.f.requires_grad_(False)
        self.f.eval()

        student = build_model_from_exp(STUDENT_EXP, 'G_ema', return_cfg=False)
        student = student.to('cuda')
        student.requires_grad_(True)
        student.train()

        opt = torch.optim.Adam(student.parameters(), lr=STUDENT_FT_LR)
        lpips = LPIPS(net='alex').to('cuda').eval()

        n_iter = 400
        for i in range(STUDENT_FT_MAX_ITER):
            opt.zero_grad()

            new_w = self.f(w, is_mtf, magnitude=random() * MAX_MAG)
            teacher_img = teacher(new_w, noise_mode='const')
            teacher_img = resize_imgs(teacher_img, STUDENT_IMSIZE)
            student_img = student(new_w.detach())

            lpips_loss = lpips(student_img, teacher_img.detach()).mean()
            pixel_loss = F.mse_loss(student_img, teacher_img.detach())
            loss = lpips_loss + pixel_loss
            early_stop = loss.item() < STUDENT_FT_EARLY_STOP

            loss.backward()
            opt.step()

            if early_stop:
                break

        print('finished finetuning')
        return student

    def _build_empty_teacher(self):
        return _Teacher(
            self.f,
            build_pretrained_sg2(load_weights=False).synthesis,
            torch.zeros((1, 18, 512), device='cuda'),
            torch.zeros((1,), device='cuda'),
        ).cuda()
