#!/usr/bin/env python3
from ai_old.dataset.dataset_base import DatasetBase
from ai_old.dataset import filter_func as ff


class FgBgFacegenDataset(DatasetBase):
    def filter_func(self):
        return ff.for_dataset(self.get_mm_dataset_name())

    def select_cols(self):
        return {
            'face_image': 'y',
            'soft_bg_mask': 'seg',
        }

    def test_set_label(self):
        return 'ffhq-test-1'

    def val_set_label(self):
        return 'ffhq-val-1'

    def get_mm_dataset_name(self):
        return 'ffhq-128'


class FgBg256FacegenDataset(DatasetBase):
    def filter_func(self):
        return ff.for_dataset(self.get_mm_dataset_name())

    def select_cols(self):
        return {
            'face_image_256': 'y',
            'soft_bg_mask_256': 'seg',
        }

    def test_set_label(self):
        return 'ffhq-test-1'

    def val_set_label(self):
        return 'ffhq-val-1'

    def get_mm_dataset_name(self):
        return 'ffhq-128'
