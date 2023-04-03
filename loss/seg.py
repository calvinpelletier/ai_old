#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from external.sg2 import training_stats


def cross_entropy2d(input, target, class_weight=None):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()
    assert h == ht and w == wt

    # input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    # target = target.view(-1)
    loss = F.cross_entropy(
        input,
        target,
        weight=class_weight,
        reduction='mean',
    )
    return loss


class SegLoss:
    def __init__(self, cfg, trainer, device):
        self.cfg = cfg
        self.trainer = trainer
        self.device = device

    def accumulate_gradients(self,
        phase,
        w,
        seg,
        sync,
        gain,
    ):
        assert phase == 'seg'

        with torch.autograd.profiler.record_function('seg_forward'):
            pred = self.trainer.run_G(w, sync)
            loss = cross_entropy2d(pred, seg)
            training_stats.report('loss/seg', loss)

        with torch.autograd.profiler.record_function('seg_backward'):
            loss.mul(gain).backward()
