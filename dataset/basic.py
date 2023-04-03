#!/usr/bin/env python3
import torch
import numpy as np
import pyspng


# just an image dataset without all the rest of asi stuff
class BasicImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths):
        self._img_paths = img_paths

        raw_shape = [len(self._img_paths)] + list(self._load_raw_image(0).shape)
        self._raw_shape = list(raw_shape)

    def __len__(self):
        return self._raw_shape[0]

    def __getitem__(self, idx):
        image = self._load_raw_image(idx)
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        return image.copy()

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    def _load_raw_image(self, idx):
        img_path = self._img_paths[idx]
        with open(img_path, 'rb') as f:
            image = pyspng.load(f.read())
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image
