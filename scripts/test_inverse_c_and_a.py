#!/usr/bin/env python3
import ai_old.constants as c
import dlib
import numpy as np
import scipy
import scipy.ndimage
from PIL import Image
from ai_old.util.face import get_faces, get_landmarks, align_face
import math
from ai_old.util.inverse import solo_aligned_to_fam_final
from ai_old.nn.models.inpaint.gated import GatedGenerator


full = Image.open('/home/asiu/data/tmp/inverse/full.jpg')
full_np = np.asarray(full)
_, w, h = full_np.shape
box = get_faces(full_np)[0]
lms = get_landmarks(full_np, box)
aligned, quad = align_face(full, lms, 128)
aligned.save('/home/asiu/data/tmp/inverse/aligned.png', 'PNG')
inpainter = GatedGenerator().cuda().eval()
fam_final = solo_aligned_to_fam_final(aligned, quad, full, inpainter, debug=True)
