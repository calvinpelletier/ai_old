#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from external.sg2 import training_stats
from ai_old.loss.hair import calc_nonhair_l2_pixel_loss
from ai_old.loss.perceptual.face import SoloFaceIdLoss
from ai_old.loss.gender import GenderLoss


def _check_if_loss_enabled(cfg, label):
    return hasattr(cfg.loss, label) and getattr(cfg.loss, label).weight > 0.


class SoloClassifyLerpLoss(nn.Module):
    def __init__(self,
        img,
        w,
        gender,
        imsize=256,
        face_weight=0.1,
        delta_weight=0.8,
        classify_weight=1.,
        use_l2_for_classify=False
    ):
        super().__init__()
        self.face_weight = face_weight
        self.delta_weight = delta_weight
        self.classify_weight = classify_weight

        self.face_loss_model = SoloFaceIdLoss(
            img).eval().requires_grad_(False)
        self.classify_loss_model = GenderLoss(
            l2=use_l2_for_classify).eval().requires_grad_(False)

        # self.register_buffer('base_img', img.clone().detach())
        self.register_buffer('base_w', w.clone().detach())
        self.register_buffer('target_gender', (1. - gender).detach())

    def forward(self, new_img, new_w):
        face_loss = self.face_loss_model(new_img)
        delta_loss = F.mse_loss(new_w, self.base_w)
        classify_loss = self.classify_loss_model(new_img, self.target_gender)
        return face_loss * self.face_weight + delta_loss * self.delta_weight + \
            classify_loss * self.classify_weight


class WPlusLerpLoss:
    def __init__(self, cfg, trainer, device):
        self.cfg = cfg
        self.trainer = trainer
        self.device = device

        self.face_loss_enabled = _check_if_loss_enabled(cfg, 'face')
        self.clip_loss_enabled = _check_if_loss_enabled(cfg, 'clip')
        self.hair_clip_loss_enabled = _check_if_loss_enabled(cfg, 'hair_clip')
        self.delta_loss_enabled = _check_if_loss_enabled(cfg, 'delta')
        self.ss_delta_loss_enabled = _check_if_loss_enabled(cfg, 'ss_delta')
        self.classify_loss_enabled = _check_if_loss_enabled(cfg, 'classify')
        self.mouth_loss_enabled = _check_if_loss_enabled(cfg, 'mouth')

    def accumulate_gradients(self,
        phase,
        img,
        gender,
        w,
        sync,
        gain,
    ):
        assert phase == 'main'

        with torch.autograd.profiler.record_function('main_forward'):
            results = self.trainer.run_G(
                    img,
                    gender,
                    w,
                    sync,
                )

            loss = 0.

            if self.face_loss_enabled:
                face_loss = results['face_loss'].mean()
                training_stats.report('loss/face', face_loss)
                loss += face_loss * self.cfg.loss.face.weight

            if self.clip_loss_enabled:
                clip_loss = results['clip_loss']
                training_stats.report('loss/clip', clip_loss)
                loss += clip_loss * self.cfg.loss.clip.weight

            if self.hair_clip_loss_enabled:
                hair_clip_loss = results['hair_clip_loss']
                training_stats.report('loss/hair_clip', hair_clip_loss)
                loss += hair_clip_loss * self.cfg.loss.hair_clip.weight

            if self.delta_loss_enabled:
                delta_loss = F.mse_loss(results['new_w'], w)
                training_stats.report('loss/delta', delta_loss)
                loss += delta_loss * self.cfg.loss.delta.weight

            if self.ss_delta_loss_enabled:
                ss_delta_loss = results['ss_delta_loss']
                training_stats.report('loss/ss_delta', ss_delta_loss)
                loss += ss_delta_loss * self.cfg.loss.ss_delta.weight

            if self.classify_loss_enabled:
                classify_loss = results['classify_loss']
                training_stats.report('loss/classify', classify_loss)
                loss += classify_loss * self.cfg.loss.classify.weight

            if self.mouth_loss_enabled:
                mouth_loss = results['mouth_loss']
                training_stats.report('loss/mouth', mouth_loss)
                loss += mouth_loss * self.cfg.loss.mouth.weight

            training_stats.report('loss/total', loss)

        with torch.autograd.profiler.record_function('main_backward'):
            loss.mul(gain).backward()


class WPlusDualLerpLoss:
    def __init__(self, cfg, trainer, device):
        self.cfg = cfg
        self.trainer = trainer
        self.device = device

        self.face_loss_enabled = _check_if_loss_enabled(cfg, 'face')
        self.clip_loss_enabled = _check_if_loss_enabled(cfg, 'clip')
        self.hair_clip_loss_enabled = _check_if_loss_enabled(cfg, 'hair_clip')
        self.delta_loss_enabled = _check_if_loss_enabled(cfg, 'delta')
        self.ss_delta_loss_enabled = _check_if_loss_enabled(cfg, 'ss_delta')
        self.nonhair_l2_pixel_loss_enabled = \
            _check_if_loss_enabled(cfg, 'nonhair_l2_pixel')

    def accumulate_gradients(self,
        phase,
        img,
        gender,
        w,
        sync,
        gain,
    ):
        assert phase == 'main'

        with torch.autograd.profiler.record_function('main_forward'):
            results = self.trainer.run_G(
                    img,
                    gender,
                    w,
                    sync,
                )

            loss = 0.

            if self.face_loss_enabled:
                face_loss = results['face_loss'].mean()
                training_stats.report('loss/face', face_loss)
                loss += face_loss * self.cfg.loss.face.weight

            if self.clip_loss_enabled:
                clip_loss = results['clip_loss']
                training_stats.report('loss/clip', clip_loss)
                loss += clip_loss * self.cfg.loss.clip.weight

            if self.hair_clip_loss_enabled:
                hair_clip_loss = results['hair_clip_loss']
                training_stats.report('loss/hair_clip', hair_clip_loss)
                loss += hair_clip_loss * self.cfg.loss.hair_clip.weight

            if self.delta_loss_enabled:
                delta_loss = F.mse_loss(results['w2'], results['w1'])
                training_stats.report('loss/delta', delta_loss)
                loss += delta_loss * self.cfg.loss.delta.weight

            if self.ss_delta_loss_enabled:
                ss_delta_loss = results['ss_delta_loss']
                training_stats.report('loss/ss_delta', ss_delta_loss)
                loss += ss_delta_loss * self.cfg.loss.ss_delta.weight

            if self.nonhair_l2_pixel_loss_enabled:
                nonhair_l2_pixel_cfg = self.cfg.loss.nonhair_l2_pixel
                nonhair_l2_pixel_loss = calc_nonhair_l2_pixel_loss(
                    results['img1'],
                    results['img2'],
                    results['seg1'],
                    results['seg2'],
                    detach_mask=nonhair_l2_pixel_cfg.detach_mask,
                    area_normalize=nonhair_l2_pixel_cfg.area_normalize,
                )
                training_stats.report(
                    'loss/nonhair_l2_pixel',
                    nonhair_l2_pixel_loss,
                )
                loss += nonhair_l2_pixel_loss * \
                    self.cfg.loss.nonhair_l2_pixel.weight

            training_stats.report('loss/total', loss)

        with torch.autograd.profiler.record_function('main_backward'):
            loss.mul(gain).backward()
