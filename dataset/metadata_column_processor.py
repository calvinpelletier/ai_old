"""
This file contains all of the logic for how we read data that's sourced from different columns in metadata.

For some columns, the metadata value will be a file path, in which case we read the file into its appropriate data
structure (likely a Torch tensor). Others may simply be inline primitives or objects, in which case no disk read is
needed.
"""
from copy import deepcopy
from inspect import signature
from os.path import join
from random import random

import ai_old.constants as c
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from ai_old.util.age import rand_age_from_range, encode_age
import pyspng


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MetadataColumnProcessor(object):
    def __init__(self, can_flip=True, override_column_procs=None):

        self.can_flip = can_flip

        # Master mapping of (column name) -> (processor)
        #
        # Processor can either be a function on the original column value, or a more robust CP object (which supports, for instance, adding columns).
        DEFAULT_COLUMN_PROCESSOR_MAP = {
            "item_id": no_op,
            "face_image": read_image,
            "face_image_256": read_image,
            "face_image_512": read_image,
            "face_image_1024": read_image,
            "segmented_face": read_np_array,
            "z": read_np_array,
            "has_hat": wrap_bool_in_tensor,
            "has_glasses": wrap_bool_in_tensor,
            "mouth_size": wrap_single_value_in_tensor,
            "gender": wrap_single_value_in_tensor,
            "age_range": CP(no_op, age=age_range_to_age),
            "dynamic_ss": read_image,
            "age_pred": no_op,
            "ss_age_pred": no_op,
            "test_set": CP.drop(),
            "set_labels": CP.drop(),
            "gender_confidence": no_op,
            "ss_gender_confidence": no_op,
            "ss_gender": wrap_single_value_in_tensor,
            "male_image": read_image,
            "female_image": read_image,
            "male_seg": read_np_array,
            "fg": read_image,
            "male": read_image,
            "female": read_image,
            "real": read_image,
            "real_fg": read_image,
            "real_ibg": read_image,
            "real_gender": wrap_single_value_in_tensor,
            "dynamic_ss_seg": read_np_array,
            "dynamic_ss_fg": read_image,
            "male_fg": read_image,
            "female_fg": read_image,
            "dilated_fg_mask": read_np_array,
            "ss_dilated_fg_mask": read_np_array,
            "ibg": read_image,
            "ss_ibg": read_image,
            "male_ibg": read_image,
            "female_ibg": read_image,
            "gaussian_seed": read_np_array,
            "real_img": read_image,
            "synth_img": read_image,
            "ss_img": read_image,
            "synth_gender": wrap_single_value_in_tensor,
            "soft_bg_mask": read_np_array,
            "soft_bg_mask_256": read_np_array,
            "clip_attrs": read_np_array,
            "e4e_inv_1024": read_image,
            "e4e_inv_256": read_image,
            "e4e_inv_128": read_image,
            "e4e_inv_swap_256": read_image,
            "e4e_inv_swap_128": read_image,
            "outer_1536": read_image,
            "outer_512": read_image,
            "outer_384": read_image,
            "outer_192": read_image,
            "e4e_inv_w_plus": read_np_array,
            "e4e_inv_swap_w_plus": read_np_array,
            "e4e_inv_clip": read_np_array,
            "fhbc_128": read_np_array,
            "e4e_inv_seg_128": read_np_array,
            "outer_fhbc_seg_384": read_np_array,
            "outer_fhbc_seg_192": read_np_array,
            "inpaint_mask_512": read_mask,
            'fg_mask_256': read_mask,
            # "enc_4x4_base": read_tensor,
            # "enc_4x4_guide": read_tensor,
            "enc_4x4_target": read_tensor,
        }

        # Apply overrides, if provided.
        if override_column_procs:
            for k, v in override_column_procs.items():
                DEFAULT_COLUMN_PROCESSOR_MAP[k] = v

        # Convert all shorthand defs (function ptrs) into actual column processor objects.
        self.column_processor_map = {}
        for k, v in DEFAULT_COLUMN_PROCESSOR_MAP.items():
            if isinstance(v, CP):
                self.column_processor_map[k] = v
            elif callable(v):
                self.column_processor_map[k] = CP(inplace_method=v)
            else:
                raise RuntimeError("Unknown type found in column proc map: %s" % str(type(v)))

    def process_row(self, row_dict):
        result = deepcopy(row_dict)

        # decide on flipping for all images in row
        hflip = self.can_flip and random() > 0.5

        for k, v in row_dict.items():
            if k not in self.column_processor_map:
                raise RuntimeError("Key %s does not have a processor" % str(k))

            self.column_processor_map[k](k, result, extra_arg=hflip)
        return result


class CP:
    def __init__(self, inplace_method, **add_new):
        self.inplace_method = inplace_method
        self.add_new = add_new

        self.include_extra_arg = False
        if self.inplace_method:
            if len(signature(self.inplace_method).parameters) == 2:
                self.include_extra_arg = True

    @staticmethod
    def drop():
        return CP(inplace_method=None)

    def __call__(self, orig_key, row_dict, extra_arg=None):
        if self.inplace_method:
            if self.include_extra_arg:
                row_dict[orig_key] = self.inplace_method(row_dict[orig_key], extra_arg)
            else:
                row_dict[orig_key] = self.inplace_method(row_dict[orig_key])
        else:
            row_dict.pop(orig_key, None)

        for new_key, val_func in self.add_new.items():
            row_dict[new_key] = val_func(row_dict[orig_key])


#####################
# Column Processors #
#####################
def read_tensor(rel_path):
    return torch.load(join(c.ASI_DATASETS_PATH, rel_path))


def read_np_array(rel_path):
    return torch.from_numpy(np.load(join(c.ASI_DATASETS_PATH, rel_path)))


# NOTE: normalization has been moved outside of column processing because
# it's faster to send the img to the gpu as uint8
def read_image(rel_path, hflip):
    img = __read_image_helper(rel_path)
    if hflip:
        img = img[:, :, ::-1]
    return img.copy()


def read_mask(rel_path, hflip):
    with open(join(c.ASI_DATASETS_PATH, rel_path), 'rb') as f:
        mask = pyspng.load(f.read())
    if hflip:
        mask = mask[:, ::-1]
    return mask.copy()


def wrap_single_value_in_tensor(val):
    return torch.tensor(val)


def wrap_bool_in_tensor(val):
    return torch.tensor(1. if val else 0.)


def age_range_to_age(age_range: str):
    lower, upper = [int(x) for x in age_range.split('-')]
    return encode_age(rand_age_from_range(lower, upper))


def no_op(val):
    return val


def read_image_no_hflip(rel_path):
    img = __read_image_helper(rel_path)
    return img.copy()


################
# Helper Funcs #
################
def __read_image_helper(rel_path):
    with open(join(c.ASI_DATASETS_PATH, rel_path), 'rb') as f:
        img = pyspng.load(f.read())
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    return img.transpose(2, 0, 1) # hwc -> chw
