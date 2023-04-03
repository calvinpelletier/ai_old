#!/usr/bin/env python3
import os
from PIL import Image
from tqdm import tqdm


def flatten(src, dest):
    os.makedirs(dest, exist_ok=True)
    for subfolder in os.listdir(src):
        for fname in os.listdir(os.path.join(src, subfolder)):
            os.rename(
                os.path.join(src, subfolder, fname),
                os.path.join(dest, fname),
            )

# flatten(
#     '/home/calvin/datasets/asdf/thumbnails128x128/',
#     '/home/calvin/datasets/ffhq-128/x/',
# )
#
# flatten(
#     '/home/calvin/datasets/asdf/thumbnails128x128_2/',
#     '/home/calvin/datasets/ffhq-128/x/',
# )

flatten(
    '/home/calvin/etc/ffhq-dataset/images1024x1024/',
    '/home/calvin/datasets/ffhq-1024/x/'
)
