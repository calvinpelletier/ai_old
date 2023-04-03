#!/usr/bin/env python3
import torch
import os
from ai_old.util.etc import normalized_tensor_to_pil_img, create_img_row, \
    resize_imgs
import external.anycost.models as ac_models
from external.anycost.models.dynamic_channel import set_uniform_channel_ratio
import ai_old.dataset.filter_func as ff
from ai_old.dataset import DatasetBase

FOLDER = '/home/asiu/data/anycost'


class Dataset(DatasetBase):
    def filter_func(self):
        return ff.for_dataset(
            self.get_mm_dataset_name(),
            additional_filter=lambda x: int(x['item_id']) < 80000,
        )

    def select_cols(self):
        return {
            'item_id': 'id',
            'e4e_inv_256': 'img',
            'e4e_inv_w_plus': 'w',
        }

    def test_set_label(self):
        return 'ffhq-test-1'

    def val_set_label(self):
        return 'ffhq-val-1'

    def get_mm_dataset_name(self):
        return 'ffhq-128'


def get_dataset():
    return Dataset(False).get_test_set(
        1, # batch size
        0, # seed
        0, # rank
        1, # num gpus
        verbose=False,
    )


def get_uniform_ac_model():
    G = ac_models.get_pretrained('generator', config='anycost-ffhq-config-f')
    G.target_res = 1024
    return G.to('cuda')


def test_anycost():
    G = get_uniform_ac_model()

    ds = get_dataset()
    for batch in ds:
        assert len(batch['id']) == 1
        id = batch['id'][0]
        print(id)

        w = batch['w'].to('cuda').to(torch.float32)
        og = batch['img'].to('cuda').to(torch.float32) / 127.5 - 1

        row = [og[0]]
        for channel_ratio in [1., 0.75, 0.5, 0.25]:
            set_uniform_channel_ratio(G, channel_ratio)
            out, _ = G(w, input_is_style=True, randomize_noise=False)
            out = resize_imgs(out, 256)
            row.append(out[0])
        row = [normalized_tensor_to_pil_img(x) for x in row]
        create_img_row(row, 256).save(os.path.join(FOLDER, f'{id}.png'))


if __name__ == '__main__':
    with torch.no_grad():
        test_anycost()





























# tmp
