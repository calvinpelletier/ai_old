#!/usr/bin/env python3
from ai_old.dataset import filter_func as ff
from ai_old.dataset import DatasetBase


class ImgGenderClassificationDataset(DatasetBase):
    def filter_func(self):
        return ff.for_dataset(self.get_mm_dataset_name())

    def select_cols(self):
        return {
            'face_image': 'img',
            'gender': 'gender',
        }

    def test_set_label(self):
        return 'ffhq-test-1'

    def val_set_label(self):
        return 'ffhq-val-1'

    def get_mm_dataset_name(self):
        return 'ffhq-128'
