#!/usr/bin/env python3
import os
from PIL import Image, ImageChops
from ai_old.util.inverse import get_outer_quad
from ai_old.util.etc import create_img_row
from ai_old.util.face import align_face, custom_align_face
import json
from collections import OrderedDict, defaultdict
import sys


# avoid "Decompressed Data Too Large" error
# PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

# JSON_SPEC_PATH = '/home/asiu/data/ffhq-dataset-v2.json'
# ALIGNED_1024_FOLDER = \
#     '/home/asiu/datasets/supplemental/face_image_1024/ffhq-128'
# INPUT_FOLDER = '/home/asiu/ffhq-dataset/in-the-wild-images'
# OUTPUT_FOLDER = '/home/asiu/datasets/supplemental/outer_1536/ffhq-128'

JSON_SPEC_PATH = '/home/calvin/nvlabs-ffhq/ffhq-dataset-v2.json'
INPUT_FOLDER = '/home/calvin/nvlabs-ffhq/in-the-wild-images'
OUTPUT_FOLDER = '/home/calvin/nvlabs-ffhq/outer_1536'

DEBUG = False
DEBUG_FOLDER = '/home/asiu/data/tmp/outer'
N_PROCESSES = 14
SKIP_EXISTING = True
SKIPS = [35680, 53177]


def outer_crop_and_align_images(json_data, process_idx):
    for item_idx, item in enumerate(json_data.values()):
        if item_idx % N_PROCESSES != process_idx:
            continue

        subdir = '%05d' % (item_idx - item_idx % 1000)
        fname = '%05d.png' % item_idx
        src_file = os.path.join(INPUT_FOLDER, subdir, fname)
        if not os.path.isfile(src_file):
            continue
        out_path = os.path.join(OUTPUT_FOLDER, fname)
        if SKIP_EXISTING and os.path.isfile(out_path):
            continue
        if item_idx in SKIPS:
            continue

        print(item_idx)

        lm = item['in_the_wild']['face_landmarks']

        full = Image.open(src_file)
        new_aligned, inner_quad = align_face(full, lm, 1024)

        # sanity
        # diff = ImageChops.difference(cur_aligned, new_aligned)
        # if diff.getbbox():
        #     combo = create_img_row([cur_aligned, new_aligned], 1024)
        #     combo.save(os.path.join(DEBUG_FOLDER, 'bad_' + fname))
        #     continue

        outer_quad = get_outer_quad(inner_quad)
        outer_imsize = 1024 + 512
        outer_aligned = custom_align_face(full, outer_quad, outer_imsize)
        outer_aligned.save(out_path)

        if DEBUG:
            cur_aligned = Image.open(os.path.join(ALIGNED_1024_FOLDER, fname))
            padded_inner = Image.new(
                'RGB',
                (outer_imsize, outer_imsize),
                'black',
            )
            padded_inner.paste(cur_aligned, (256, 256))
            combo = create_img_row([padded_inner, outer_aligned], outer_imsize)
            combo.save(os.path.join(DEBUG_FOLDER, fname))


def run():
    assert len(sys.argv) == 2
    process_idx = int(sys.argv[1])
    assert process_idx < N_PROCESSES

    print('parsing json metadata...')
    with open(JSON_SPEC_PATH, 'rb') as f:
        json_data = json.load(f, object_pairs_hook=OrderedDict)

    outer_crop_and_align_images(json_data, process_idx)


if __name__ == '__main__':
    run()
