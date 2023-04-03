#!/usr/bin/env python3
from ai_old.dataset import filter_func as ff
from ai_old.dataset.dataset_base import DatasetBase


class UltDataset(DatasetBase):
    def filter_func(self):
        return ff.for_dataset(self.get_mm_dataset_name())

    def select_cols(self):
        return {
            'real_img': 'real_img',
            'real_gender': 'real_gender',
            'synth_img': 'x_img',
            'synth_gender': 'x_gender',
            'ss_img': 'y_img',
            'ss_gender': 'y_gender',
        }

    def test_set_label(self):
        return 'ult-test-1'

    def val_set_label(self):
        return 'ult-val-1'

    def get_mm_dataset_name(self):
        return 'ult-128'


# class SwapOnlyUltDataset(UltDataset):
#     def select_cols(self):
#         return {
#             'male': 'male',
#             'female': 'female',
#         }
#
#
# class FgUltDataset(UltDataset):
#     def select_cols(self):
#         return {
#             'real_fg': 'real',
#             'real_gender': 'real_gender',
#             'male_fg': 'male',
#             'female_fg': 'female',
#         }
#
#
# class BlendUltDataset(UltDataset):
#     def select_cols(self):
#         return {
#             'real': 'real',
#             'real_fg': 'real_fg',
#             'real_ibg': 'real_ibg',
#             'real_gender': 'real_gender',
#             'male': 'male',
#             'female': 'female',
#             'male_fg': 'male_fg',
#             'female_fg': 'female_fg',
#             'male_ibg': 'male_ibg',
#             'female_ibg': 'female_ibg',
#         }
#
#
# class SsOnlyBlendUltDataset(UltDataset):
#     def select_cols(self):
#         return {
#             'male': 'male',
#             'female': 'female',
#             'male_fg': 'male_fg',
#             'female_fg': 'female_fg',
#             'male_ibg': 'male_ibg',
#             'female_ibg': 'female_ibg',
#         }
