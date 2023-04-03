#!/usr/bin/env python3
from ai_old.dataset.dataset_base import DatasetBase
from ai_old.dataset import filter_func as ff


class FfhqDataset(DatasetBase):
    def filter_func(self):
        return ff.for_dataset(self.get_mm_dataset_name())

    def select_cols(self):
        return {'face_image': 'x'}

    def test_set_label(self):
        return 'ffhq-test-1'

    def val_set_label(self):
        return 'ffhq-val-1'

    def get_mm_dataset_name(self):
        return 'ffhq-128'


class FemaleFfhqDataset(FfhqDataset):
    def filter_func(self):
        return ff.for_dataset(
            'ffhq-128',
            additional_filter=lambda x: x['gender'] == 0.,
        )

    def test_set_label(self):
        return 'femaleffhq-test-1'

    def val_set_label(self):
        return 'femaleffhq-val-1'


class BlendFfhqDataset(FfhqDataset):
    def select_cols(self):
        return {
            'face_image': 'full',
            'fg': 'fg',
            'ibg': 'ibg',
        }


class FfhqGanDataset(FfhqDataset):
    def select_cols(self):
        return {'face_image': 'y'}


class Ffhq256GanDataset(FfhqDataset):
    def select_cols(self):
        return {'face_image_256': 'y'}


class FfhqFemaleGanDataset(FemaleFfhqDataset):
    def select_cols(self):
        return {'face_image': 'y'}


class FgFfhqGanDataset(FfhqGanDataset):
    def select_cols(self):
        return {'fg': 'y'}


class FgMaskFfhq256Dataset(DatasetBase):
    def filter_func(self):
        return ff.for_dataset(
            self.get_mm_dataset_name(),
            additional_filter=lambda x: int(x['item_id']) < 80000,
        )

    def select_cols(self):
        return {
            'face_image_256': 'img',
            'fg_mask_256': 'mask',
        }

    def test_set_label(self):
        return 'ffhq-test-1'

    def val_set_label(self):
        return 'ffhq-val-1'

    def get_mm_dataset_name(self):
        return 'ffhq-128'
