#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from external.sg2 import training_stats


class InversionLoss:
    def __init__(self, cfg, trainer, device):
        self.trainer = trainer
        self.device = device

    def accumulate_gradients(self, phase, gen_z, sync, gain):
        assert phase == 'enc'

        with torch.autograd.profiler.record_function('enc_forward'):
            gen_w, enc_w = self.trainer.run_G(gen_z, sync=sync)
            loss = F.mse_loss(gen_w, enc_w)
            training_stats.report('loss/enc', loss)

        with torch.autograd.profiler.record_function('enc_backward'):
            loss.mul(gain).backward()
