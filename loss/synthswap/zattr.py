#!/usr/bin/env python3
import torch.nn as nn
from ai_old.loss import ComboLoss


class ZAttrLoss(ComboLoss):
    def __init__(self, loss_conf):
        super().__init__(loss_conf)
        self.create_subloss(
            'mouth',
            nn.MSELoss(),
            ('pred_mouth', 'mouth_size'),
        )
        self.create_subloss(
            'glasses',
            nn.BCEWithLogitsLoss(),
            ('pred_glasses', 'has_glasses'),
        )
