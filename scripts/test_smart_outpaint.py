#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import cv2
import ai_old.dataset.filter_func as ff
from ai_old.dataset import DatasetBase
from ai_old.nn.models.seg.seg import binary_seg_to_img, colorize
from ai_old.util.factory import build_model_from_exp
from ai_old.util.etc import create_img_row, create_img_row, \
    normalized_tensor_to_pil_img
from ai_old.util.outer import get_di_k, get_outer_boundary_mask


def get_dataset(imsize):
    class Dataset(DatasetBase):
        def filter_func(self):
            return ff.for_dataset(
                self.get_mm_dataset_name(),
                additional_filter=lambda x: int(x['item_id']) < 80000,
            )

        def select_cols(self):
            return {
                'item_id': 'id',
                f'outer_{imsize}': 'img',
                f'outer_fhbc_seg_{imsize}': 'seg',
            }

        def test_set_label(self):
            return 'ffhq-test-1'

        def val_set_label(self):
            return 'ffhq-val-1'

        def get_mm_dataset_name(self):
            return 'ffhq-128'

    return Dataset(False).get_test_set(
        1, # batch size
        0, # seed
        0, # rank
        1, # num gpus
        verbose=False,
    )


def seg_test():
    imsize = 192
    di_k = get_di_k(imsize)

    outer_seg_predictor = build_model_from_exp(
        'outer-seg/0/1',
        'model',
        return_cfg=False,
    ).to('cuda')
    outer_seg_predictor.eval()
    assert not outer_seg_predictor.pred_from_seg_only

    outer_boundary_mask = get_outer_boundary_mask(imsize).to('cuda')
    inv_outer_boundary_mask = 1. - outer_boundary_mask

    ds = get_dataset(imsize)
    for batch in ds:
        id = batch['id'][0]
        print(id)

        seg = batch['seg'].to('cuda').to(torch.long)
        img = batch['img'].to('cuda').to(torch.float32) / 127.5 - 1

        seg_pred = outer_seg_predictor(seg, img)[0]
        seg_pred = F.softmax(seg_pred, dim=0)

        face = seg[0] == 0
        hair = seg[0] == 1
        facehair = torch.bitwise_or(face, hair).float()
        # facehair = seg_pred[0, :, :] + seg_pred[1, :, :]
        # facehair = (facehair > 0.5).float()
        inner_gan_mask = facehair * outer_seg_predictor.inner_mask

        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (di_k, di_k))
        dilated_facehair = cv2.dilate(
            facehair.cpu().numpy() * 255., dilate_kernel)
        dilated_facehair = torch.tensor(dilated_facehair / 255.).to('cuda')
        dilated_facehair = (dilated_facehair > 0.5).float()
        inv_inner_gan_mask = 1. - inner_gan_mask
        inpaint_mask = dilated_facehair * inv_inner_gan_mask * \
            inv_outer_boundary_mask

        gt_mask = torch.ones_like(inpaint_mask) * (1. - inpaint_mask) * \
            inv_inner_gan_mask

        combo = torch.zeros_like(gt_mask).to(torch.long)
        for y in range(imsize):
            for x in range(imsize):
                assert (inner_gan_mask[y][x] + inpaint_mask[y][x] + \
                    gt_mask[y][x]) == 1.
                if inner_gan_mask[y][x] == 1.:
                    combo[y][x] = 1
                elif inpaint_mask[y][x] == 1.:
                    combo[y][x] = 2
                else:
                    pass # ... = 0

        row = [
            normalized_tensor_to_pil_img(img[0]),
            colorize(seg, needs_argmax=False)[0],
            colorize(seg_pred.unsqueeze(0))[0],
            binary_seg_to_img(inner_gan_mask).convert('RGB'),
            binary_seg_to_img(outer_boundary_mask).convert('RGB'),
            binary_seg_to_img(inpaint_mask).convert('RGB'),
            binary_seg_to_img(gt_mask).convert('RGB'),
            colorize(combo.unsqueeze(0), needs_argmax=False)[0],
        ]
        create_img_row(row, imsize).save(
            f'/home/asiu/data/tmp/outpaint/{id}.png')


if __name__ == '__main__':
    with torch.no_grad():
        seg_test()




















# tmp
