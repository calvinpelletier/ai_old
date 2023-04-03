#!/usr/bin/env python3
import torch.nn as nn
from ai_old.loss import ComboLoss, SumLoss
from ai_old.nn.models.synthswap.zattr import ZAttrPredictor
from ai_old.util.factory import build_model_from_exp
from ai_old.loss.coral import CoralSubloss


class DynamicGenderLerpLoss(ComboLoss):
    def __init__(self, loss_conf, is_for_eval=False):
        super().__init__(loss_conf, is_for_eval=is_for_eval)
        self.create_subloss(
            'mag',
            SumLoss(),
            ('adj_mag',),
        )
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
        self.create_subloss(
            'age',
            CoralSubloss(),
            ('z_age_pred', 'scaled_age_enc')
        )

        self.attr_predictor = build_model_from_exp(
            'z-attr-pred/0/14',
            freeze=True,
        )
        self.attr_predictor.eval()

        self.age_predictor = build_model_from_exp(
            'z-age-pred/1/7',
            freeze=True,
        )
        self.age_predictor.eval()

    def forward(self, ents):
        attr_preds = self.attr_predictor(ents['z2'])
        age_pred = self.age_predictor(ents['z2'])
        return self.calc_and_sum_sublosses({**ents, **attr_preds, **age_pred})
