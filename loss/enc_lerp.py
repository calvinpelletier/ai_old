#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from external.sg2 import training_stats


class EncLerpLoss:
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
            pred_enc, target_enc = self.trainer.run_model(
                img,
                gender,
                w,
                sync,
            )

            loss = F.mse_loss(pred_enc, target_enc.detach())
            # loss = torch.mean(torch.abs(pred_enc - target_enc.detach()))
            training_stats.report('loss/total', loss)

        with torch.autograd.profiler.record_function('main_backward'):
            loss.mul(gain).backward()


class FastEncLerpLoss:
    def __init__(self, cfg, trainer, device):
        self.cfg = cfg
        self.trainer = trainer
        self.device = device

    def accumulate_gradients(self, phase, target_enc, sync, gain):
        assert phase == 'main'
        with torch.autograd.profiler.record_function('main_forward'):
            pred_enc = self.trainer.run_model(target_enc, sync)
            loss = F.mse_loss(pred_enc, target_enc.detach())
            training_stats.report('loss/total', loss)
        with torch.autograd.profiler.record_function('main_backward'):
            loss.mul(gain).backward()
