#!/usr/bin/env python3
from ai_old.util.etc import binary_acc
from ai_old.util.metrics_writer import MetricsWriter
import os
from torchvision.utils import save_image

# NOTE:
# validation (val): running the training loss functions on the val dataset
# evaluation (eval) task-specific metrics independent of individual experiments


class Task:
    def __init__(self, trainer, results_path):
        self._init_val_metrics(trainer, results_path)
        self._init_eval_metrics(results_path)

    def _init_val_metrics(self, trainer, results_path):
        if trainer.loss.has_multiple_losses():
            self.trainer_has_multiple_losses = True
            metric_names = ['total_loss', 'weighted_total_loss']
            metric_names += sorted(trainer.loss.get_loss_names())
        else:
            self.trainer_has_multiple_losses = False
            metric_names = ['loss']

        self.val_metrics = {metric: 0. for metric in metric_names}
        self.val_metrics_writer = MetricsWriter(
            results_path,
            metric_names,
            'val',
        )

    def _init_eval_metrics(self, results_path):
        metric_names = self._get_eval_metric_names()
        self.eval_metrics = {x: 0. for x in metric_names}
        if len(self.eval_metrics):
            self.eval_metrics_writer = MetricsWriter(
                results_path,
                metric_names,
                'eval',
            )
        else:
            self.eval_metrics_writer = None

    def _get_eval_metric_names(self):
        return []

    def begin_val_eval(self, total_steps):
        self.total_steps = total_steps

        for k in self.val_metrics:
            self.val_metrics[k] = 0.

        for k in self.eval_metrics:
            self.eval_metrics[k] = 0.

    def val_eval_step(self, data, out, loss, sublosses):
        # IMPORTANT: val must be calculated before eval because eval often
        # reuses val results
        self._val_step(data, out, loss, sublosses)
        self._eval_step(data, out)

    def sampling_step(self, data, out, step):
        pass

    def finish_val_eval(self, val_dataset_len):
        self._finish_val(val_dataset_len)
        self._finish_eval(val_dataset_len)

    def train_step(self, out, step, eval_freq):
        pass

    def _val_step(self, data, out, loss, sublosses):
        if self.trainer_has_multiple_losses:
            self.val_metrics['weighted_total_loss'] += loss.cpu().item()
            for l in sublosses:
                self.val_metrics[l] += sublosses[l].item()
                self.val_metrics['total_loss'] += sublosses[l].item()
        else:
            self.val_metrics['loss'] += loss.cpu().item()

    def _eval_step(self, data, out):
        pass

    def _finish_val(self, val_dataset_len):
        for k in self.val_metrics:
            self.val_metrics[k] /= val_dataset_len
        self.val_metrics_writer.write_metric(
            self.total_steps,
            self.val_metrics,
        )

    def _finish_eval(self, val_dataset_len):
        pass

    def requires_img_manifold(self):
        return False


# TODO: needs to be redone with new metrics system
# class BinaryClassificationTask(Task):
#     def __init__(self, trainer, results_path):
#         metric_names = ['loss', 'acc']
#         self.metrics = {metric: 0. for metric in metric_names}
#
#         self.metrics_writer = MetricsWriter(results_path, metric_names)
#
#         # TODO: replace with something better
#         self.pred_key = trainer.loss.pred_key
#         self.target_key = trainer.loss.target_key
#
#     def eval_step(self, data, out, loss, _sublosses):
#         self.metrics['loss'] += loss.cpu().item()
#         self.metrics['acc'] += binary_acc(
#             out[self.pred_key],
#             data[self.target_key],
#         ).cpu().item()
