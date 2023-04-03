#!/usr/bin/env python3
from ai_old.dataset import filter_func as ff
from ai_old.dataset import DatasetBase
import ai_old.dataset.metadata_column_processor as mcp
from ai_old.util import age


class GaussianSeedDataset(DatasetBase):
    def filter_func(self):
        return ff.for_dataset('gaussian-512')

    def select_cols(self):
        return {'gaussian_seed': 'z'}

    def test_set_label(self):
        return 'gaussian-512-test'

    def val_set_label(self):
        return 'gaussian-512-val'

    def get_mm_dataset_name(self):
        return 'gaussian-512'

    def has_training_set(self):
        return False
