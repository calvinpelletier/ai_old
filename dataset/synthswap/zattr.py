#!/usr/bin/env python3
from ai_old.dataset import DatasetBase
from ai_old.dataset import filter_func as ff


class ZAttrDataset(DatasetBase):
    def filter_func(self):
        return ff.for_dataset("facegen")

    def select_cols(self):
        return {"z": "z", "mouth_size": "mouth_size", "has_glasses": "has_glasses"}
