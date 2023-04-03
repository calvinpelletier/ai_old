#!/usr/bin/env python3
from ai_old.dataset import filter_func as ff
from ai_old.dataset import DatasetBase
import ai_old.dataset.metadata_column_processor as mcp
from ai_old.util import age


class ImgAgeClassificationDataset(DatasetBase):
    def filter_func(self):
        return ff.for_dataset('ffhq-128')

    def select_cols(self):
        return ['face_image', 'age_range']


class ZAgeDataset(DatasetBase):
    def filter_func(self):
        return ff.for_dataset('facegen')

    def select_cols(self):
        return ['z', 'age_pred']

    def test_set_label(self):
        return 'facegen1'

    def get_column_processor_overrides(self):
        return {
            'age_pred': mcp.CP(
                inplace_method=mcp.no_op,
                age_enc=age.encode_age,
            )
        }


class ZScaledAgeDataset(DatasetBase):
    def filter_func(self):
        return ff.for_dataset('facegen')

    def select_cols(self):
        return ['z', 'age_pred']

    def test_set_label(self):
        return 'facegen1'

    def get_column_processor_overrides(self):
        return {
            'age_pred': mcp.CP(
                inplace_method=mcp.no_op,
                scaled_age_enc=age.scale_and_encode_age,
            )
        }


class ImgScaledAgeDataset(DatasetBase):
    def filter_func(self):
        return ff.for_dataset(
            self.get_mm_dataset_name(),
            additional_filter=lambda x: int(x['item_id']) < 80000,
        )

    def select_cols(self):
        return {
            'face_image_256': 'img',
            'age_pred': 'age_pred',
        }

    def test_set_label(self):
        return 'ffhq-test-1'

    def val_set_label(self):
        return 'ffhq-val-1'

    def get_column_processor_overrides(self):
        return {
            'age_pred': mcp.CP(
                inplace_method=mcp.no_op,
                scaled_age_enc=age.scale_and_encode_age,
            )
        }

    def get_mm_dataset_name(self):
        return 'ffhq-128'
