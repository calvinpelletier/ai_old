#!/usr/bin/env python3
import torch
import ai_old.dataset.filter_func as ff
from ai_old.dataset import DatasetBase
import matplotlib.pyplot as plt
from ai_old.loss.clip import ManualClipLoss
from ai_old.util.etc import normalized_tensor_to_pil_img


class Dataset(DatasetBase):
    def filter_func(self):
        return ff.for_dataset(
            self.get_mm_dataset_name(),
            additional_filter=lambda x: int(x['item_id']) < 80000,
        )

    def select_cols(self):
        return {
            'item_id': 'id',
            'face_image_256': 'img',
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


def run():
    ds = get_dataset()
    clip_loss = ManualClipLoss('cuda')
    text_encs = clip_loss.prep_texts(['long hair', 'short hair'])
    # long_vals = []
    # short_vals = []
    data = []
    for batch in ds:
        id = batch['id'][0]
        img = batch['img'].to('cuda').to(torch.float32) / 127.5 - 1
        long = clip_loss(img, text_encs[0]).item()
        short = clip_loss(img, text_encs[1]).item()
        print(id, long, short)
        # long_vals.append(long)
        # short_vals.append(short)
        data.append((img[0], long - short))
    data = sorted(data, key=lambda x: x[1])
    for i, (img, val) in enumerate(data):
        normalized_tensor_to_pil_img(img).save(
            f'/home/asiu/data/tmp/attr/{i:05d}.png')

    # plt.scatter(long_vals, short_vals)
    # plt.savefig('/home/asiu/data/tmp/attr/tmp.png')
    # plt.clf()


if __name__ == '__main__':
    with torch.no_grad():
        run()
