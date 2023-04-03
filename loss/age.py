#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_old.util.factory import legacy_build_model_from_exp


class AgeDeltaLoss(nn.Module):
    def __init__(self, base_img):
        super().__init__()

        self.age_classifier = legacy_build_model_from_exp(
            'age-pred/0/0', verbose=False).eval().to('cuda')

        target_age = self.age_classifier(base_img)['age_pred']
        self.register_buffer('target_age', target_age.detach())

    def forward(self, img):
        age = self.age_classifier(img)['age_pred']
        return F.mse_loss(age, self.target_age).mean()
