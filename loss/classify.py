#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from external.sg2 import training_stats


class BinaryClassificationLoss:
    def __init__(self, cfg, trainer, device):
        self.trainer = trainer
        self.device = device

    def accumulate_gradients(self, phase, img, gender, sync, gain):
        assert phase == 'classify'

        with torch.autograd.profiler.record_function('classify_forward'):
            pred = self.trainer.run_C(img, sync=sync)
            loss = F.binary_cross_entropy_with_logits(pred, gender)
            training_stats.report('loss/bce', loss)

        with torch.autograd.profiler.record_function('classify_backward'):
            loss.mul(gain).backward()
