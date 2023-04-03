#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_old.nn.models.seg.seg import Segmenter, LABEL_TO_IDX, binary_seg_to_img
from ai_old.util.etc import resize_imgs, create_img_row, normalized_tensor_to_pil_img


# TMP_COUNT = 0


def calc_nonhair_l2_pixel_loss(
    img1,
    img2,
    seg1,
    seg2,
    detach_mask=True,
    area_normalize=False,
):
    n = img1.shape[0]
    imsize = img1.shape[2]
    assert img1.shape == (n, 3, imsize, imsize)
    assert img2.shape == (n, 3, imsize, imsize)
    assert seg1.shape == (n, 4, imsize, imsize)
    assert seg2.shape == (n, 4, imsize, imsize)
    hair_idx = 1

    seg1 = F.softmax(seg1, dim=1)
    seg2 = F.softmax(seg2, dim=1)
    seg1_nonhair = 1. - seg1[:, hair_idx, :, :]
    seg2_nonhair = 1. - seg2[:, hair_idx, :, :]
    nonhair_union = seg1_nonhair * seg2_nonhair
    nonhair_union = nonhair_union.unsqueeze(dim=1)

    if detach_mask:
        nonhair_union = nonhair_union.detach()

    masked1 = img1 * nonhair_union
    masked2 = img2 * nonhair_union

    # tmp
    # global TMP_COUNT
    # if TMP_COUNT % 100 == 0:
    #     tmp = [
    #         binary_seg_to_img(seg1_nonhair[0]),
    #         binary_seg_to_img(seg2_nonhair[0]),
    #         binary_seg_to_img(nonhair_union[0][0]),
    #     ]
    #     tmp = create_img_row(tmp, 256, mode='L')
    #     tmp.save(f'/home/asiu/data/tmp/dual-lerp/seg_{TMP_COUNT // 100}.png')
    #
    #     tmp = [
    #         normalized_tensor_to_pil_img(img1[0]),
    #         normalized_tensor_to_pil_img(masked1[0]),
    #         normalized_tensor_to_pil_img(img2[0]),
    #         normalized_tensor_to_pil_img(masked2[0]),
    #     ]
    #     tmp = create_img_row(tmp, 256)
    #     tmp.save(f'/home/asiu/data/tmp/dual-lerp/img_{TMP_COUNT // 100}.png')
    # TMP_COUNT += 1

    loss = F.mse_loss(masked1, masked2)

    if area_normalize:
        loss /= torch.sum(nonhair_union)

    return loss


class NonHairPixelLoss(nn.Module):
    def __init__(self, base_img):
        super().__init__()
        b, c, h, w = base_img.shape
        assert c == 3
        assert h == w
        imsize = h

        self.segmenter = Segmenter(imsize=imsize, label=None).to('cuda')
        self.hair_idx = LABEL_TO_IDX['hair']
        self.hat_idx = LABEL_TO_IDX['hat']

        base_seg = self.segmenter(base_img)
        base_nonhair_mask = 1. - (base_seg[:, self.hair_idx, :, :] + \
            base_seg[:, self.hat_idx, :, :])
        assert base_nonhair_mask.shape == (b, imsize, imsize)
        self.register_buffer('base_nonhair_mask', base_nonhair_mask.detach())
        self.register_buffer('base_img', base_img.detach())

    def forward(self, img):
        seg = self.segmenter(img)
        hair_mask = (seg[:, self.hair_idx, :, :] + \
            seg[:, self.hat_idx, :, :])
        nonhair_mask = 1. - hair_mask

        nonhair_union = nonhair_mask * self.base_nonhair_mask
        nonhair_union = nonhair_union.unsqueeze(dim=1)

        masked = img * nonhair_union
        base_masked = self.base_img * nonhair_union

        loss = F.mse_loss(masked, base_masked.detach())
        loss = loss / torch.sum(nonhair_union)
        return loss
