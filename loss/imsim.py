#!/usr/bin/env python3
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS
from ai_old.loss.perceptual.face import FaceIdLoss, SoloFaceIdLoss
from ai_old.loss.perceptual.trad import SoloPerceptualLoss


class FaceImSimLoss(nn.Module):
    def __init__(self,
        imsize,
        l2_weight=1.0,
        lpips_weight=0.8,
        face_weight=0.1,
    ):
        super().__init__()
        self.l2_weight = l2_weight
        self.lpips_weight = lpips_weight
        self.face_weight = face_weight

        self.face_loss_model = FaceIdLoss(imsize).eval()
        self.lpips_loss_model = LPIPS(net='alex').eval()

    def forward(self, a, b):
        l2_loss = F.mse_loss(a, b)
        lpips_loss = self.lpips_loss_model(a, b).mean()
        face_loss = self.face_loss_model(a, b).mean()
        return face_loss * self.face_weight + l2_loss * self.l2_weight + \
            lpips_loss * self.lpips_weight


class SoloFaceImSimLoss(nn.Module):
    def __init__(self,
        imsize,
        target,
        l2_weight=1.0,
        percep_weight=0.8,
        face_weight=0.1,
    ):
        super().__init__()
        self.l2_weight = l2_weight
        self.percep_weight = percep_weight
        self.face_weight = face_weight

        self.face_loss_model = SoloFaceIdLoss(imsize, target).eval()
        # self.percep_loss_model = SoloPerceptualLoss(target).eval()
        self.percep_loss_model = LPIPS(net='alex').eval()

        self.register_buffer('target', target.clone().detach())

    def forward(self, x):
        l2_loss = F.mse_loss(x, self.target.detach())
        # percep_loss = self.percep_loss_model(x).mean()
        percep_loss = self.percep_loss_model(x, self.target.detach()).mean()
        face_loss = self.face_loss_model(x)
        return face_loss * self.face_weight + l2_loss * self.l2_weight + \
            percep_loss * self.percep_weight


# class ImsimLoss(ComboLoss):
#     def __init__(self, conf):
#         super().__init__(conf)
#
#         # l2 pixel loss
#         if hasattr(conf, 'pixel'):
#             self.create_subloss(
#                 'pixel',
#                 nn.MSELoss(),
#                 ('g_fake', 'y'),
#             )
#
#         # perceptual loss (either traditional perceptual or lpips)
#         if hasattr(conf, 'perceptual'):
#             assert not hasattr(conf, 'lpips'), 'probably a mistake'
#             self.create_subloss(
#                 'perceptual',
#                 PerceptualLoss(),
#                 ('g_fake', 'y'),
#             )
#         if hasattr(conf, 'lpips'):
#             assert not hasattr(conf, 'perceptual'), 'probably a mistake'
#             self.create_subloss(
#                 'lpips',
#                 LpipsLoss(),
#                 ('g_fake', 'y'),
#             )
#
#         # face loss
#         if hasattr(conf, 'face'):
#             self.create_subloss(
#                 'face',
#                 FaceLoss(),
#                 ('g_fake', 'y'),
#             )
