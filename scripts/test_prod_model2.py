#!/usr/bin/env python3
from unit.prod import get_prod_model
from ai_old.server.inferencer import get_cropped_and_aligned_face
import numpy as np
import torch
from torchvision.transforms import transforms
from PIL import Image
from torchvision.utils import save_image
from collections import namedtuple

FaceImageInferenceInput = namedtuple("FaceImageInferenceInput", "full_image bounding_box")
img = FaceImageInferenceInput(Image.open('/home/asiu/data/tmp/inverse/full.jpeg'), {
        "h": 0.3134765625,
        "w": 0.3134765625,
        "x": 0.2705078125,
        "y": 0.166015625
})
imgs = [img]

img_to_tensor = transforms.Compose([transforms.ToTensor()])
prod_model = get_prod_model(exp='blend-ult/1/0')
face_im_batch = torch.stack(
    [img_to_tensor(get_cropped_and_aligned_face(np.asarray(img.full_image), img.bounding_box)) for img in
     imgs]).cuda()
print(face_im_batch.shape)
save_image(
    face_im_batch,
    '/home/asiu/data/tmp/inverse/asdf2.png',
    normalize=False,
    # range=(-1, 1),
)
results = prod_model(face_im_batch, is_mtf=True, debug=True)
save_image(
    results['rec'],
    '/home/asiu/data/tmp/inverse/asdf.png',
    normalize=True,
    range=(-1, 1),
)
