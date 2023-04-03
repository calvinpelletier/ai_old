#!/usr/bin/env python3
import torch
from external.sg2 import training_stats


class LerpGenLoss:
    def __init__(self, cfg, trainer, device):
        self.cfg = cfg
        self.trainer = trainer
        self.device = device

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
            delta_loss, face_loss, classify_loss, reg_loss = \
                self.trainer.run_G(
                    img,
                    gender,
                    w,
                    sync,
                )

            training_stats.report('loss/face', face_loss)
            training_stats.report('loss/delta', delta_loss)
            training_stats.report('loss/classify', classify_loss)
            training_stats.report('loss/reg', reg_loss)
            loss = face_loss * self.cfg.loss.face.weight + \
                delta_loss * self.cfg.loss.delta.weight + \
                classify_loss * self.cfg.loss.classify.weight + \
                reg_loss * self.cfg.loss.reg.weight
            training_stats.report('loss/total', loss)

        with torch.autograd.profiler.record_function('main_backward'):
            loss.mul(gain).backward()
