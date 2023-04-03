#!/usr/bin/env python3
import os
import numpy as np
from torch.autograd import Variable
import torch
import time
import cv2
from external.dpr.utils.utils_SH import *
from external.dpr.model import *
import ai_old.constants as c
from tqdm import tqdm

# ---------------- create normal for rendering half sphere ------
img_size = 256
x = np.linspace(-1, 1, img_size)
z = np.linspace(1, -1, img_size)
x, z = np.meshgrid(x, z)

mag = np.sqrt(x**2 + z**2)
valid = mag <=1
y = -np.sqrt(1 - (x*valid)**2 - (z*valid)**2)
x = x * valid
y = y * valid
z = z * valid
normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
normal = np.reshape(normal, (-1, 3))
#-----------------------------------------------------------------

img_res = 256
model_res = 512
model_path = os.path.join(c.PRETRAINED_MODELS, 'dpr', 'trained_model_03.t7')
example_light = os.path.join(
    c.ASI_CODE_PATH,
    'dpr/example_light/rotate_light_00.txt',
)
debug_mode = False
base_dir = os.path.join(c.ASI_DATASETS_PATH, 'e4e-facegen-9-1')
img_in_dir = os.path.join(base_dir, 'imgs')
img_out_dir = os.path.join(base_dir, 'debug/relit-00')
lighting_out_dir = os.path.join(base_dir, 'lighting')
light_sphere_dir = os.path.join(base_dir, 'debug/light-sphere')

# load model
model = HourglassNet()
model.load_state_dict(torch.load(model_path))
model.cuda()
model.train(False)

for fname in tqdm(sorted(os.listdir(img_in_dir))):
    img = cv2.imread(os.path.join(img_in_dir, fname))
    row, col, _ = img.shape
    img = cv2.resize(img, (512, 512))
    Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    inputL = Lab[:,:,0]
    inputL = inputL.astype(np.float32)/255.0
    inputL = inputL.transpose((0,1))
    inputL = inputL[None,None,...]
    inputL = Variable(torch.from_numpy(inputL).cuda())

    sh = np.loadtxt(example_light)
    sh = sh[0:9]
    sh = sh * 0.7

    #----------------------------------------------
    #  rendering images using the network
    sh = np.reshape(sh, (1,9,1,1)).astype(np.float32)
    sh = Variable(torch.from_numpy(sh).cuda())
    outputImg, outputSH  = model(inputL, sh, 0)
    if debug_mode:
        outputImg = outputImg[0].cpu().data.numpy()
        outputImg = outputImg.transpose((1,2,0))
        outputImg = np.squeeze(outputImg)
        outputImg = (outputImg*255.0).astype(np.uint8)
        Lab[:,:,0] = outputImg
        resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
        resultLab = cv2.resize(resultLab, (col, row))
        cv2.imwrite(os.path.join(img_out_dir, fname), resultLab)
    #----------------------------------------------

    #--------------------------------------------------
    # rendering half-sphere
    sh = np.squeeze(outputSH.cpu().detach().numpy())
    np.save(os.path.join(lighting_out_dir, fname.split('.')[0] + '.npy'), sh)
    if debug_mode:
        shading = get_shading(normal, sh)
        value = np.percentile(shading, 95)
        ind = shading > value
        shading[ind] = value
        shading = (shading - np.min(shading))/(np.max(shading) - np.min(shading))
        shading = (shading *255.0).astype(np.uint8)
        shading = np.reshape(shading, (256, 256))
        shading = shading * valid
        cv2.imwrite(os.path.join(light_sphere_dir, fname), shading)
    #--------------------------------------------------
