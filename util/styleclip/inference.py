#!/usr/bin/env python3
from ai_old.util.styleclip.manipulate import Manipulator
import numpy as np
import torch
import clip
from ai_old.util.styleclip.map_ts import get_boundary, get_dt


class StyleClip():
    def __init__(self, dataset_name='ffhq'):
        self.model, _ = clip.load('ViT-B/32', device='cuda', jit=False)
        self.load_data(dataset_name)


    def load_data(self, dataset_name):
        self.M = Manipulator(dataset_name=dataset_name)

        self.fs3 = np.load(
            f'/home/asiu/data/styleclip/npy/{dataset_name}/fs3.npy')

        self.ws = np.load(
            f'/home/asiu/data/styleclip/inf/w_plus.npy')
        self.ws = self.ws[4]
        self.ws = torch.tensor(self.ws).to('cuda').unsqueeze(dim=0)
        self.M.stylespace = self.M.w_to_style(self.ws)

        self.c_threshold = 20
        self.set_init_p()


    def set_init_p(self):
        self.M.alpha = [3]
        self.M.num_images = 1

        self.target = ''
        self.neutral = ''
        self.get_dt2()
        img_index = 0
        self.M.stylespace_tmp = [
            tmp[img_index:(img_index+1)] for tmp in self.M.stylespace
        ]


    def get_dt2(self):
        classnames = [self.target, self.neutral]
        dt = get_dt(classnames, self.model)

        self.dt = dt
        num_cs = []
        betas = np.arange(0.1, 0.3, 0.01)
        for i in range(len(betas)):
            boundary_tmp2, num_c = get_boundary(
                self.fs3,
                self.dt,
                self.M,
                threshold=betas[i],
            )
            print(betas[i])
            num_cs.append(num_c)

        num_cs = np.array(num_cs)
        select = num_cs > self.c_threshold

        if sum(select) == 0:
            self.beta = 0.1
        else:
            self.beta = betas[select][-1]


    def get_code(self):
        boundary_tmp2, num_c = get_boundary(
            self.fs3,
            self.dt,
            self.M,
            threshold=self.beta,
        )
        codes = self.M.ms_code(self.M.stylespace_tmp, boundary_tmp2)
        return codes


    def get_img(self):
        codes = self.get_code()
        out = self.M.generate_img(self.ws, codes)
        img = out[0, 0]
        return img
