#!/usr/bin/env python3
from ai_old.dataset.dataset_base import DatasetBase
from ai_old.dataset import filter_func as ff


class SegFromGenDataset(DatasetBase):
    def filter_func(self):
        return ff.for_dataset(self.get_mm_dataset_name())

    def select_cols(self):
        return {
            'e4e_inv_w_plus': 'w',
            'fhbc_128': 'seg',
        }

    def test_set_label(self):
        return 'ffhq-test-1'

    def val_set_label(self):
        return 'ffhq-val-1'

    def get_mm_dataset_name(self):
        return 'ffhq-128'


class FullSegFromGenDataset(DatasetBase):
    def filter_func(self):
        return ff.for_dataset(
            self.get_mm_dataset_name(),
            additional_filter=lambda x: int(x['item_id']) < 80000,
        )

    def select_cols(self):
        return {
            'e4e_inv_w_plus': 'w',
            'e4e_inv_seg_128': 'seg',
        }

    def test_set_label(self):
        return 'ffhq-test-1'

    def val_set_label(self):
        return 'ffhq-val-1'

    def get_mm_dataset_name(self):
        return 'ffhq-128'
