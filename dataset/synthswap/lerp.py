#!/usr/bin/env python3
import ai_old.dataset.metadata_column_processor as mcp
from ai_old.dataset import DatasetBase
from ai_old.dataset import filter_func as ff
from ai_old.util import age


class GenderLerpDataset(DatasetBase):
    def filter_func(self):
        return ff.for_dataset(self.get_mm_dataset_name())

    def select_cols(self):
        return {'z': 'z',
                'gender': 'gender',
                'mouth_size': 'mouth_size',
                'has_glasses': 'has_glasses',
                'z_age': 'z_age'}

    def test_set_label(self):
        return 'facegen-test-gender-lerp'

    def val_set_label(self):
        return 'facegen-val-gender-lerp'

    def get_column_processor_overrides(self):
        return {
            'z_age': mcp.CP(
                inplace_method=mcp.no_op,
                scaled_age_enc=age.scale_and_encode_age,
            )
        }

    def get_mm_dataset_name(self):
        return 'facegen'
