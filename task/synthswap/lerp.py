#!/usr/bin/env python3
import os
from ai_old.task import Task
from ai_old.nn.models.facegen import FaceGenerator
from torchvision.utils import save_image
from ai_old.loss.synthswap.lerp import DynamicGenderLerpLoss


class SynthSwapTask(Task):
    def __init__(self, trainer, results_path):
        super().__init__(trainer, results_path)
        # samples
        samples_path = os.path.join(results_path, 'samples')
        self.samples_path = os.path.join(samples_path, 'step_{}_im_{}.png')
        os.makedirs(os.path.dirname(self.samples_path))

        # facegen model
        self.facegen = FaceGenerator(imsize=128).to('cuda')
        self.facegen.eval()

        # eval loss function if needed
        if self.__has_eval_metric_not_in_val():
            self.eval_loss = DynamicGenderLerpLoss(
                None, is_for_eval=True).to('cuda')
        else:
            self.eval_loss = None

    def _get_eval_metric_names(self):
        return ['age', 'glasses', 'mag', 'mouth']

    def __has_eval_metric_not_in_val(self):
        for eval_metric in self._get_eval_metric_names():
            if eval_metric not in self.val_metrics:
                return True
        return False

    def _eval_step(self, data, out):
        sublosses = None
        if self.eval_loss is not None:
            _, sublosses, _ = self.eval_loss({**data, **out})

        for metric in self.eval_metrics:
            if metric in self.val_metrics:
                self.eval_metrics[metric] = self.val_metrics[metric]
            else:
                self.eval_metrics[metric] += sublosses[metric].item()

    def _finish_eval(self, val_dataset_len):
        for k in self.eval_metrics:
            self.eval_metrics[k] /= val_dataset_len
        self.eval_metrics_writer.write_metric(
            self.total_steps,
            self.eval_metrics,
        )

    def sampling_step(self, data, out, step):
        for i, (item_id, z2) in enumerate(zip(data['item_id'], out['z2'])):
            im = self.facegen(
                z2.unsqueeze(0),  # stylegan expects (n, 512)
                is_entangled=False,
            ).squeeze()

            save_image(
                im,
                self.samples_path.format(step, item_id),
                normalize=True,
                range=(-1, 1),
            )

    def finish_eval(self, val_len, step):
        super().finish_eval(val_len, step)
