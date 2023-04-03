#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from external.sg2 import training_stats
from ai_old.loss.seg import cross_entropy2d


class OuterSegLoss:
    def __init__(self, cfg, trainer, device):
        self.cfg = cfg
        self.trainer = trainer
        self.device = device

    def accumulate_gradients(self,
        phase,
        seg,
        img,
        sync,
        gain,
    ):
        assert phase == 'main'

        with torch.autograd.profiler.record_function('main_forward'):
            pred = self.trainer.run_model(seg, img, sync)
            loss = cross_entropy2d(pred, seg)
            training_stats.report('loss/seg', loss)

        with torch.autograd.profiler.record_function('main_backward'):
            loss.mul(gain).backward()
