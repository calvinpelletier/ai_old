#!/usr/bin/env python3
import torch
import torch.nn as nn
from ai_old.loss.base import BaseLoss


# used by fake trainers for debugging purposes
class NullLoss(BaseLoss):
    def __init__(self, _conf):
        super().__init__()
        self.loss = torch.tensor(0.)

    def forward(self, _ents):
        return self.loss, None, None
