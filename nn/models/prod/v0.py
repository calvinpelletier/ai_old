#!/usr/bin/env python3
import torch
from ai_old.nn.models import StaticUnit
from ai_old.util.factory import build_model_from_exp
import numpy as np
import cv2
import torchvision.transforms as transforms
from ai_old.nn.models.inpaint.gated import GatedGenerator
from ai_old.nn.models.seg import Segmenter


MASK_DILATE_KERNEL_SIZE = 16


'''
input args:
    face: normalized image tensor (n x 3 x 128 x 128)
    is_mtf: tensor of 1s and 0s (n) (1 = male-to-female, 0 = female-to-male)
    debug: bool (if true, generate images)
output dict:
    ibg: inner inpainted background (same shape as face input)
    identity: float tensor (n x ? x 4 x 4)
    z_src: float tensor (n x ?)
    z_dest: same as z_src
    [debug] rec: generator output using z_src
    [debug] swap: generator output using z_dest
'''
class ProdV0(StaticUnit):
    def __init__(self, exp):
        super().__init__()

        # image transforms
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.blur = transforms.GaussianBlur(7, sigma=1.)

        # segmentation/masking
        self.seg = Segmenter(imsize=128)
        self.mask_dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (MASK_DILATE_KERNEL_SIZE, MASK_DILATE_KERNEL_SIZE),
        )

        # background inpainting
        self.inpainter = GatedGenerator()

        # main
        # self.ult = build_model_from_exp(exp)

    def forward(self, face, is_mtf, debug=False):
        bs = face.shape[0]

        # normalize face image
        normed_face = self.normalize(face)

        # seg
        seg = self.seg(normed_face, colorize=False)
        seg = torch.argmax(seg, dim=1, keepdim=True)

        # seg to mask (this works because 0 is the background and all labels >=1
        # are the foreground)
        fg_mask = torch.clamp(seg, 0., 1.)
        fg_mask = self.blur(fg_mask)

        # extract foreground
        fg = normed_face * fg_mask

        # dilate foreground mask (to make sure the inpainter doesnt glimpse a
        # strand of hair and try to generate a face)
        dilated_fg_mask = fg_mask.cpu().numpy() * 255.
        for i in range(dilated_fg_mask.shape[0]):
            dilated_fg_mask[i] = cv2.dilate(
                dilated_fg_mask[i],
                self.mask_dilate_kernel,
            )
        dilated_fg_mask = torch.from_numpy(np.uint8(dilated_fg_mask) / 255)
        dilated_fg_mask = dilated_fg_mask.to('cuda').float()

        # inpaint background
        bg_mask = 1. - dilated_fg_mask
        ibg = self.inpainter(face, dilated_fg_mask)
        ibg = face * bg_mask + ibg * dilated_fg_mask
        ibg = self.normalize(ibg)

        # encode the foreground
        z_src = self.ult.e_pri(fg)
        identity = self.ult.e_mod(fg, z_src)

        # predict the ideal destination z
        z_dest = self.ult.t(z_src, is_mtf)

        ret = {
            'ibg': ibg,
            'identity': identity,
            'z_src': z_src,
            'z_dest': z_dest,
        }

        if debug:
            # recreate the original image
            ret['rec'] = self.ult.g(identity, z_src, ibg)

            # generate a full gender swap
            ret['swap'] = self.ult.g(identity, z_dest, ibg)

        return ret

    def finalize(self, ibg, identity, z_src, z_dest, slider):
        z = z_src + (z_dest - z_src) * slider
        return self.ult.g(identity, z, ibg)
