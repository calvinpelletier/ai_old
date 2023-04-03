#!/usr/bin/env python3
import torch
from external.sg2 import training_stats


class RealOnlyClipIdSwapLoss:
    def __init__(self, cfg, trainer, device):
        self.cfg = cfg
        self.trainer = trainer
        self.device = device

    def accumulate_gradients(self,
        phase,
        real_img,
        real_gender,
        real_w,
        sync,
        gain,
    ):
        assert phase == 'swap'

        with torch.autograd.profiler.record_function('swap_forward'):
            clip_loss, face_loss, delta = \
                self.trainer.run_G(
                    real_img,
                    real_gender,
                    real_w,
                    sync,
                )

            if self.cfg.loss.delta.ord == 1:
                delta_loss = delta.abs()
            elif self.cfg.loss.delta.ord == 2:
                delta_loss = delta.square()
            else:
                raise Exception(self.cfg.loss.delta.ord)

            training_stats.report('loss/clip', clip_loss)
            training_stats.report('loss/face', face_loss)
            training_stats.report('loss/delta', delta_loss)

            loss = clip_loss.mean() * self.cfg.loss.clip.weight + \
                face_loss.mean() * self.cfg.loss.id.weight + \
                delta_loss.mean() * self.cfg.loss.delta.weight

            training_stats.report('loss/total', loss)

        with torch.autograd.profiler.record_function('swap_backward'):
            loss.mul(gain).backward()
