#!/usr/bin/env python3
from ai_old.dataset import filter_func as ff
from ai_old.dataset.dataset_base import DatasetBase


class PairedGenderDataset(DatasetBase):
    def filter_func(self):
        return ff.for_dataset(
            self.get_mm_dataset_name(),
            # Not all facegen images will have these. They're only set if gender
            # and ss_gender are successfully made different.
            additional_filter=lambda x: \
                'male_image' in x and 'female_image' in x,
        )

    def select_cols(self):
        return {
            'male_image': 'x',
            'female_image': 'y',
        }

    def test_set_label(self):
        return 'pg-test-1'

    def val_set_label(self):
        return 'pg-val-1'

    def get_mm_dataset_name(self):
        return 'facegen'


class SegPairedGenderDataset(PairedGenderDataset):
    def select_cols(self):
        return {
            'male_image': 'x',
            'female_image': 'y',
            'male_seg': 'seg',
        }
