#!/usr/bin/env python3
from ai_old.dataset.dataset_base import DatasetBase
from ai_old.dataset import filter_func as ff


class SwapDataset(DatasetBase):
    def filter_func(self):
        return ff.for_dataset(
            self.get_mm_dataset_name(),

            # check that face_image and dynamic_ss are different genders
            # (see scripts/md/add_male_and_female_cols)
            additional_filter=lambda x: \
                'male_image' in x and 'female_image' in x,
        )

    def select_cols(self):
        return {
            'face_image': 'x_img',
            'gender': 'x_gender',
            'dynamic_ss': 'y_img',
            'ss_gender': 'y_gender',
        }

    def test_set_label(self):
        return 'pg-test-1'

    def val_set_label(self):
        return 'pg-val-1'

    def get_mm_dataset_name(self):
        return 'facegen'


class BlendSwapDataset(DatasetBase):
    def filter_func(self):
        return ff.for_dataset(
            self.get_mm_dataset_name(),

            # check that face_image and dynamic_ss are different genders
            # (see scripts/md/add_male_and_female_cols)
            additional_filter=lambda x: \
                'male_image' in x and 'female_image' in x,
        )

    def select_cols(self):
        return {
            'face_image': 'x_img',
            'fg': 'x_fg',
            'ibg': 'x_ibg',
            'gender': 'x_gender',
            'dynamic_ss': 'y_img',
            'dynamic_ss_fg': 'y_fg',
            'ss_ibg': 'y_ibg',
            'ss_gender': 'y_gender',
        }

    def test_set_label(self):
        return 'pg-test-1'

    def val_set_label(self):
        return 'pg-val-1'

    def get_mm_dataset_name(self):
        return 'facegen'
