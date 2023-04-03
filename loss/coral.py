#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_old.loss.base import BaseLoss


# pred: (bs, n_classes)
# target: (bs, n_classes)
# class_weights: (n_classes) or None
class CoralLoss(BaseLoss):
    def __init__(self, loss_conf):
        super().__init__()
        self.pred_key = loss_conf.pred_key
        self.target_key = loss_conf.target_key
        self.class_weights = None # TODO

    def forward(self, ents):
        pred = ents[self.pred_key]
        target = ents[self.target_key]
        assert pred.shape == target.shape, f'{pred.shape} != {target.shape}'

        x = F.logsigmoid(pred) * target + \
            (F.logsigmoid(pred) - pred) * (1 - target)

        if self.class_weights is not None:
            x *= self.class_weights

        # loss, sublosses, subloss_times
        return torch.mean(-torch.sum(x, dim=1)), None, None


# TODO: reduce duplicated code with CoralLoss
class CoralSubloss(nn.Module):
    def forward(self, pred, target):
        assert pred.shape == target.shape, f'{pred.shape} != {target.shape}'

        x = F.logsigmoid(pred) * target + \
            (F.logsigmoid(pred) - pred) * (1 - target)

        return torch.mean(-torch.sum(x, dim=1))
