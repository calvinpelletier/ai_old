#!/usr/bin/env python3
import torch
from PIL import Image
from ai_old.util.face import get_faces, get_landmarks, align_face
import numpy as np
from ai_old.util.etc import normalized_tensor_to_pil_img, create_img_grid
from external.sg2.augment import AugmentPipe


OUTPUT_PATH = '/home/asiu/data/tmp/aug.png'


def align(full_path, face_idx):
    img = Image.open(full_path)
    img_np = np.asarray(img)
    box = get_faces(img_np)[face_idx]
    landmarks = get_landmarks(img_np, box)
    aligned, inner_quad = align_face(img, landmarks, 1024)
    return img, aligned, inner_quad


def pil_to_tensor(img):
    img_np = np.asarray(img).transpose(2, 0, 1)
    img_tensor = torch.from_numpy(np.copy(img_np))
    img_tensor = img_tensor.to('cuda').to(torch.float32) / 127.5 - 1
    return img_tensor.unsqueeze(0)


def run(full_path, face_idx):
    full_img, aligned, inner_quad = align(full_path, face_idx)
    aligned = aligned.resize((256, 256), Image.LANCZOS)
    img_tensor = pil_to_tensor(aligned)
    img_tensor = img_tensor.repeat(8, 1, 1, 1)

    aug = AugmentPipe(
        brightness=0.5,
        contrast=0.5,
        lumaflip=0,
        hue=0.25,
        saturation=0.5,
        brightness_std=0.1,
        contrast_std=0.25,
        hue_max=0.05,
        saturation_std=0.25,
    )

    grid = []
    for i in range(8):
        grid.append(
            [normalized_tensor_to_pil_img(img) for img in aug(img_tensor)])
    create_img_grid(grid, 256).save(OUTPUT_PATH)


if __name__ == '__main__':
    run('/home/asiu/data/sera/og/12.jpg', 0)
