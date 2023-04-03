#!/usr/bin/env python3
import torch
import torch.nn as nn
import os
import ai_old.constants as c
import numpy as np


class BaseSwapper(nn.Module):
    def __init__(self, gender_dir_path='mtf/0.npy'):
        super().__init__()

        gender_dir = torch.tensor(np.load(
            os.path.join(c.LERP_PATH, gender_dir_path),
        )).float()
        self.register_buffer('gender_dir', gender_dir)

    def forward(self, w, src_gender, magnitude=1.):
        sign = src_gender * 2. - 1. # 1=male,0=female -> 1=male,-1=female
        delta = sign * self.gender_dir * magnitude
        new_w = w + delta
        return new_w, delta


class ConstAgeSwapper(nn.Module):
    def __init__(self,
        gender_dir_path='mtf/0.npy',
        age_dir_path='age/1.gz',
    ):
        super().__init__()

        gender_dir = torch.tensor(np.load(
            os.path.join(c.LERP_PATH, gender_dir_path),
        )).float()
        self.register_buffer('gender_dir', gender_dir)

        age_dir = torch.tensor(np.loadtxt(
            os.path.join(c.LERP_PATH, age_dir_path),
        )).float()
        self.register_buffer('age_dir', age_dir)

        self.age_mag = nn.Parameter(torch.tensor(0.))

    def forward(self, w, src_gender, magnitude=1.):
        sign = src_gender * 2. - 1. # 1=male,0=female -> 1=male,-1=female
        dir = self.gender_dir + self.age_dir * self.age_mag
        delta = sign * dir * magnitude
        new_w = w + delta
        return new_w, delta
