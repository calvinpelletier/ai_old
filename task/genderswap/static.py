#!/usr/bin/env python3
from ai_old.task.manifold import ImgManifoldTask
from ai_old.loss.perceptual.lpips import LpipsLoss
from ai_old.loss.perceptual.trad import PerceptualLoss
from ai_old.util.eval import ssim, possibly_redundant_eval_metric_calc
import torch.nn.functional as F
from ai_old.nn.models.encode.arcface import ArcFaceWrapper


# TODO: clean up this file
# TODO: create a metrics tuple of (
#   name,
#   fn,
#   whether it needs /= val_size,
#   whether it needs to transfer a model to/from gpu,
#   whether it is a slow metric
# )
# TODO: don't need to copy duplicated eval metric from val every step


class StaticMtfTask(ImgManifoldTask):
    def __init__(self, trainer, results_path):
        super().__init__(trainer, results_path)
        self.__init_eval_models()

    def __init_eval_models(self):
        self.models = {}

        # lpips
        if self.__is_eval_but_not_val_metric('lpips'):
            self.models['lpips'] = LpipsLoss().eval().cpu()

        # perceptual
        if self.__is_eval_but_not_val_metric('perceptual'):
            self.models['perceptual'] = PerceptualLoss().eval().cpu()

        # arcface_diff
        if self.__is_eval_but_not_val_metric('arcface_diff'):
            self.models['arcface_diff'] = ArcFaceWrapper().eval().cpu()

    def __transfer_eval_models(self, device):
        self.models = {k: v.to(device) for k, v in self.models.items()}

    def _get_eval_metric_names(self):
        return self.__get_master_metric_names() + \
            self.__get_imsim_metric_names() + \
            self.__get_face_metric_names() + \
            self._get_manifold_metric_names()

    def __get_master_metric_names(self):
        return ['master', 'master_realism', 'master_accuracy']

    def __get_imsim_metric_names(self):
        return ['l2_pixel', 'ssim', 'lpips', 'perceptual']

    def __get_face_metric_names(self):
        return ['arcface_diff']

    def begin_val_eval(self, total_steps):
        super().begin_val_eval(total_steps)
        self.__transfer_eval_models('cuda')

    def _eval_step(self, data, out):
        super()._eval_step(data, out)
        self.__calc_mse_metric(data, out, 'l2_pixel')
        self.__calc_perceptual_metric(data, out, 'perceptual')
        self.__calc_perceptual_metric(data, out, 'lpips')
        self.__calc_ssim_metric(data, out, 'ssim')
        self.__calc_arcface_diff_metric(data, out, 'arcface_diff')

    def _finish_eval(self, val_dataset_len):
        # return models to cpu
        self.__transfer_eval_models('cpu')

        # finalize metrics calculated via val outputs
        for k in self.eval_metrics:
            if self.eval_metrics[k] is not None:
                self.eval_metrics[k] /= val_dataset_len

        self._calc_manifold_metrics()

        self.__calc_master_metrics()

        # write results to disk
        self.eval_metrics_writer.write_metric(
            self.total_steps,
            self.eval_metrics,
        )

    def __calc_master_metrics(self):
        self.eval_metrics['master_accuracy'] = \
            (0.5 - self.eval_metrics['l2_pixel']) * 5. / 0.45 + \
            (self.eval_metrics['ssim'] - 0.25) * 5. / 0.5 + \
            (0.55 - self.eval_metrics['lpips']) * 40. / 0.4 + \
            (1.17 - self.eval_metrics['perceptual']) * 10. / 0.7 + \
            (0.9 - self.eval_metrics['arcface_diff']) * 40. / 0.65

        if self.eval_metrics['fid'] is not None and \
                self.eval_metrics['manifold_precision'] is not None and \
                self.eval_metrics['manifold_recall'] is not None:
            self.eval_metrics['master_realism'] = \
                (120. - self.eval_metrics['fid']) * 90. / 80. + \
                self.eval_metrics['manifold_recall'] * 5. / 0.09 + \
                (self.eval_metrics['manifold_precision'] - 0.63) * 5. / 0.1
            self.eval_metrics['master'] = \
                (self.eval_metrics['master_realism'] + \
                self.eval_metrics['master_accuracy']) / 2.
        else:
            self.eval_metrics['master_realism'] = None
            self.eval_metrics['master'] = None

    def __is_eval_but_not_val_metric(self, metric):
        return metric in self.eval_metrics and metric not in self.val_metrics

    @possibly_redundant_eval_metric_calc
    def __calc_mse_metric(self, data, out, metric):
        self.eval_metrics[metric] += F.mse_loss(
            out[self._get_fake_key()],
            data[self._get_real_key()],
        ).item()

    @possibly_redundant_eval_metric_calc
    def __calc_perceptual_metric(self, data, out, metric):
        self.eval_metrics[metric] += self.models[metric](
            out[self._get_fake_key()], data[self._get_real_key()]).item()

    @possibly_redundant_eval_metric_calc
    def __calc_ssim_metric(self, data, out, metric):
        self.eval_metrics[metric] += ssim(
            out[self._get_fake_key()],
            data[self._get_real_key()],
        )

    @possibly_redundant_eval_metric_calc
    def __calc_arcface_diff_metric(self, data, out, metric):
        embeddings1 = self.models[metric](out[self._get_fake_key()])
        embeddings2 = self.models[metric](data[self._get_real_key()])
        bs = out[self._get_fake_key()].shape[0]
        diff = 0.
        for i in range(bs):
            cosine_similarity = embeddings1[i].dot(embeddings2[i]).item()
            diff += 1. - cosine_similarity
        self.eval_metrics[metric] += diff / bs
