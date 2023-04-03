#!/usr/bin/env python3
import torch
from ai_old.task import Task
import os
from ai_old.util.age import unscale_age

N_SAMPLES = 32


class AgeClassification(Task):
    def __init__(self, trainer, results_path):
        super().__init__(trainer, results_path)
        self.samples_path = os.path.join(results_path, 'samples.csv')

    def eval_test_step(self, data, out, step):
        with open(self.samples_path, 'w') as f:
            for id, age_range, age_pred in zip(
                data['item_id'],
                data[self.gt_key()],
                out[self.pred_key()],
            ):
                probabilities = torch.sigmoid(age_pred)
                age_pred_label = torch.sum(probabilities > 0.5).item()
                if self.is_scaled_age():
                    age_pred_label = unscale_age(age_pred_label)
                f.write(f'{id}, {age_range},{age_pred_label}\n')

    def gt_key(self):
        return 'age_range'

    def pred_key(self):
        return 'age_pred'

    def is_scaled_age(self):
        return False


class ZAgeClassification(AgeClassification):
    def gt_key(self):
        return 'age_pred'

    def pred_key(self):
        return 'z_age_pred'


class ZScaledAgeClassification(ZAgeClassification):
    def is_scaled_age(self):
        return True
