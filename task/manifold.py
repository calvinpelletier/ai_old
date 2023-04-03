#!/usr/bin/env python3
import os
from ai_old.task import Task
from torchvision.utils import save_image
from ai_old.util.precision_recall import IPR
from ai_old.util.fid import calculate_fid_given_paths
import gc
import shutil


FID_BATCH_SIZE = 32


class ImgManifoldTask(Task):
    def __init__(self, trainer, results_path):
        super().__init__(trainer, results_path)
        self.__init_paths(results_path)

    def __init_paths(self, results_path):
        # samples
        samples_path = os.path.join(results_path, 'samples')
        self.samples_path = os.path.join(samples_path, 'step_{}_im_{}.png')
        os.makedirs(os.path.dirname(self.samples_path), exist_ok=True)

        # ephemeral images for manifold metrics
        self.ephemeral_folder = os.path.join(results_path, 'ephemeral')
        self.ephemeral_path = os.path.join(self.ephemeral_folder, '{}.png')
        os.makedirs(self.ephemeral_folder, exist_ok=True)

        # manifold paths set later by trainer wrapper
        self.manifold_paths = None
        self.manifold_approx = None

    def __load_manifold_approx(self):
        self.manifold_approx = IPR()
        self.manifold_approx.load_ref(self.manifold_paths['approx'])

    def __unload_manifold_approx(self):
        if self.manifold_approx is not None:
            del self.manifold_approx
            gc.collect()
            self.manifold_approx = None

    def _get_eval_metric_names(self):
        return self._get_manifold_metric_names()

    def _get_manifold_metric_names(self):
        return [
            'manifold_realism',
            'manifold_precision',
            'manifold_recall',
            'fid',
        ]

    def __get_face_metric_names(self):
        return ['arcface_diff']

    def begin_val_eval(self, total_steps):
        super().begin_val_eval(total_steps)
        if total_steps >= self._get_min_steps_slow_metrics():
            self.__load_manifold_approx()

    def sampling_step(self, data, out, step):
        for id, generated in zip(data['item_id'], out[self._get_fake_key()]):
            save_image(
                generated,
                self.samples_path.format(step, id),
                normalize=True,
                range=(-1, 1),
            )

    def train_step(self, out, step, eval_freq):
        next_eval_at = max(
            self._get_min_steps_slow_metrics(),
            (1 + step // eval_freq) * eval_freq,
        )
        steps_till_next_eval = next_eval_at - step
        if steps_till_next_eval <= self._get_n_ephemeral():
            for i in range(out[self._get_fake_key()].shape[0]):
                save_image(
                    out[self._get_fake_key()][i],
                    self.ephemeral_path.format(step + i),
                    normalize=True,
                    range=(-1, 1),
                )

    def _eval_step(self, data, out):
        self.__calc_manifold_realism(data, out, 'manifold_realism')

    def _calc_manifold_metrics(self):
        self.__calc_manifold_p_and_r('manifold_precision', 'manifold_recall')
        self.__unload_manifold_approx() # to free up ram before fid calc
        self.__calc_fid('fid')

        # clean up
        shutil.rmtree(self.ephemeral_folder)
        os.makedirs(self.ephemeral_folder)

    def __calc_manifold_realism(self, data, out, metric):
        if self.total_steps >= self._get_min_steps_slow_metrics():
            bs = out[self._get_fake_key()].shape[0]
            realism = 0.
            for i in range(bs):
                realism += self.manifold_approx.realism(
                    out[self._get_fake_key()][i])
            self.eval_metrics[metric] += realism / bs
        else:
            self.eval_metrics[metric] = None

    def __calc_manifold_p_and_r(self, p_metric_name, r_metric_name):
        if self.total_steps >= self._get_min_steps_slow_metrics():
            print('[EVAL] calculating recision and recall...')
            pr = self.manifold_approx.precision_and_recall(
                self.ephemeral_folder)
            self.eval_metrics[p_metric_name] = pr.precision
            self.eval_metrics[r_metric_name] = pr.recall
        else:
            self.eval_metrics[p_metric_name] = None
            self.eval_metrics[r_metric_name] = None

    def __calc_fid(self, metric_name):
        if self.total_steps >= self._get_min_steps_slow_metrics():
            print('[EVAL] calculating FID...')
            self.eval_metrics[metric_name] = calculate_fid_given_paths(
                (self.manifold_paths['stats'], self.ephemeral_folder),
                FID_BATCH_SIZE,
            )
        else:
            self.eval_metrics[metric_name] = None

    def requires_img_manifold(self):
        return True

    def set_img_manifold_paths(self, paths):
        self.manifold_paths = paths

    # minimum number of steps before we begin calculating slow metrics
    def _get_min_steps_slow_metrics(self):
        return 100000

    # number of output images to save during training for manifold metrics
    def _get_n_ephemeral(self):
        return 10000

    def _get_fake_key(self):
        return 'g_fake'

    def _get_real_key(self):
        return 'y'
