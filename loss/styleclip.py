#!/usr/bin/env python3
import torch.nn as nn
import torch.nn.functional as F
from ai_old.loss.perceptual.face import SoloFaceIdLoss
from ai_old.loss.clip import ClipLoss


class SoloStyleClipLoss(nn.Module):
    def __init__(self, target_text, base_img, base_w):
        super().__init__()

        self.face_loss_model = SoloFaceIdLoss(
            base_img).eval().requires_grad_(False).to('cuda')

        self.clip_loss_model = ClipLoss(
            target_text,
            'cuda',
        ).eval().requires_grad_(False).to('cuda')

        self.register_buffer('base_w', base_w.clone().detach())

        self.face_loss_weight = 0.1
        self.clip_loss_weight = 1.
        self.delta_loss_weight = 0.8

    def forward(self, new_img, new_w):
        face_loss = self.face_loss_model(new_img).mean()
        clip_loss = self.clip_loss_model(new_img).mean()
        delta_loss = F.mse_loss(new_w, self.base_w)
        return face_loss * self.face_loss_weight + \
            clip_loss * self.clip_loss_weight + \
            delta_loss * self.delta_loss_weight
