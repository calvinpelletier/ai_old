#!/usr/bin/env python3
import os
import ai_old.constants as c
import numpy as np
import torch
from PIL import Image
from ai_old.server.common import timing, log
from ai_old.util import face
from torchvision.transforms import transforms
from unit.inpaint.gated import GatedGenerator
from unit.prod import get_prod_model
from util.inverse import solo_aligned_to_fam_final

PROD_MODEL_BASE = 'ai_old.nn.models.prod.v1.ProdV1'
PROD_MODEL_EXP = None

img_to_tensor = transforms.Compose([transforms.ToTensor()])

IMSIZE = 128

prod_model = get_prod_model(base=PROD_MODEL_BASE, exp=PROD_MODEL_EXP)

aligned_face_ims, quad_values = zip(*[
    get_cropped_and_aligned_face(
        np.asarray(img.full_image),
        img.bounding_box,
    ) for img in imgs])
face_im_batch = torch.stack([
    img_to_tensor(img) for img in aligned_face_ims]).cuda()

is_mtf_batch = torch.FloatTensor([img.is_mtf for img in imgs]).cuda()

results = prod_model(face_im_batch, is_mtf_batch, debug=True)

for i in range(num_images):
    print(results['w_plus'][i].shape)
