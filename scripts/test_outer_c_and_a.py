#!/usr/bin/env python3
import ai_old.constants as c
import dlib
import numpy as np
import scipy
import scipy.ndimage
from PIL import Image
from ai_old.util.face import get_faces, get_landmarks, align_face, \
    custom_align_face
import math
from ai_old.util.inverse import get_outer_quad
from ai_old.util.etc import create_img_row


full = Image.open('/home/asiu/data/tmp/inverse/full.jpg')
full_np = np.asarray(full)
_, w, h = full_np.shape
box = get_faces(full_np)[0]
lms = get_landmarks(full_np, box)
aligned, inner_quad = align_face(full, lms, 1024)
aligned.save('/home/asiu/data/tmp/inverse/aligned.png', 'PNG')

outer_quad = get_outer_quad(inner_quad, full=full, debug=True)
outer_imsize = 1024 + 512
outer_aligned = custom_align_face(full, outer_quad, outer_imsize)
outer_aligned.save('/home/asiu/data/tmp/inverse/outer_aligned.png', 'PNG')

padded_inner = Image.new(
    'RGB',
    (outer_imsize, outer_imsize),
    'black',
)
padded_inner.paste(aligned, (256, 256))
combo = create_img_row([padded_inner, outer_aligned], outer_imsize)
combo.save('/home/asiu/data/tmp/inverse/combo.png')
