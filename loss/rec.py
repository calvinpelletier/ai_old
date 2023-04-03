#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from external.sg2 import training_stats


class CustomRecLoss:
    def __init__(self, cfg, trainer, device):
        self.trainer = trainer
        self.device = device

        if hasattr(cfg.loss, 'perceptual'):
            self.perceptual_weight = cfg.loss.perceptual.weight
        else:
            self.perceptual_weight = None

        if hasattr(cfg.loss, 'face'):
            self.face_weight = cfg.loss.face.weight
        else:
            self.face_weight = None

    def accumulate_gradients(self, phase, real_img, sync, gain):
        assert phase == 'rec'

        with torch.autograd.profiler.record_function('rec_forward'):
            gen_img, face_loss, perceptual_loss = self.trainer.run_G(
                real_img,
                sync=sync,
            )
            pixel_loss = F.mse_loss(gen_img, real_img)
            training_stats.report('loss/pixel', pixel_loss)
            loss = pixel_loss.mean()

            if self.face_weight is not None:
                training_stats.report('loss/face', face_loss)
                loss += face_loss.mean() * self.face_weight

            if self.perceptual_weight is not None:
                training_stats.report('loss/perceptual', perceptual_loss)
                loss += perceptual_loss.mean() * self.perceptual_weight

            training_stats.report('loss/weighted_total', loss)

        with torch.autograd.profiler.record_function('rec_backward'):
            loss.mul(gain).backward()


class FancyRecLoss:
    def __init__(self, cfg, trainer, device):
        self.trainer = trainer
        self.device = device

        # TODO: make configurable
        self.perceptual_weight = 0.8
        self.face_weight = 0.1

    def accumulate_gradients(self, phase, real_img, sync, gain):
        assert phase == 'rec'

        with torch.autograd.profiler.record_function('rec_forward'):
            gen_img, face_loss, perceptual_loss = self.trainer.run_G(
                real_img,
                sync=sync,
            )
            pixel_loss = F.mse_loss(gen_img, real_img)
            training_stats.report('loss/pixel', pixel_loss)
            training_stats.report('loss/face', face_loss)
            training_stats.report('loss/perceptual', perceptual_loss)
            loss = pixel_loss + \
                perceptual_loss * self.perceptual_weight + \
                face_loss * self.face_weight

        with torch.autograd.profiler.record_function('rec_backward'):
            loss.mul(gain).backward()


class E4eFancyRecLoss:
    def __init__(self, cfg, trainer, device):
        self.trainer = trainer
        self.device = device

        # TODO: make configurable
        self.perceptual_weight = 0.8
        self.face_weight = 0.1

        self.delta_weight = cfg.loss.delta_weight

    def accumulate_gradients(self, phase, real_img, sync, gain):
        assert phase == 'rec'

        with torch.autograd.profiler.record_function('rec_forward'):
            gen_img, delta_loss, face_loss, percep_loss = self.trainer.run_G(
                real_img,
                sync=sync,
            )
            pixel_loss = F.mse_loss(gen_img, real_img)

            training_stats.report('loss/pixel', pixel_loss)
            training_stats.report('loss/face', face_loss)
            training_stats.report('loss/perceptual', percep_loss)
            training_stats.report('loss/delta', delta_loss)
            loss = pixel_loss + \
                percep_loss * self.perceptual_weight + \
                face_loss * self.face_weight + \
                delta_loss * self.delta_weight

        with torch.autograd.profiler.record_function('rec_backward'):
            loss.mul(gain).backward()


class RestyleFancyRecLoss:
    def __init__(self, cfg, trainer, device):
        self.trainer = trainer
        self.device = device

        # TODO: make configurable
        self.perceptual_weight = 0.8
        self.face_weight = 0.1
        self.n_iters = 5

    def accumulate_gradients(self, phase, real_img, sync, gain):
        assert phase == 'rec'
        ws = None
        img = None
        for i in range(self.n_iters):
            with torch.autograd.profiler.record_function(f'rec_forward{i}'):
                img, ws, face_loss, perceptual_loss = self.trainer.run_G(
                    real_img,
                    img,
                    ws,
                    sync=sync,
                )
                pixel_loss = F.mse_loss(img, real_img)
                training_stats.report(f'loss/pixel{i}', pixel_loss)
                training_stats.report(f'loss/face{i}', face_loss)
                training_stats.report(f'loss/perceptual{i}', perceptual_loss)
                loss = pixel_loss + \
                    perceptual_loss * self.perceptual_weight + \
                    face_loss * self.face_weight

            with torch.autograd.profiler.record_function(f'rec_backward{i}'):
                loss.mul(gain).backward()


class BlendFancyRecLoss:
    def __init__(self, cfg, trainer, device):
        self.trainer = trainer
        self.device = device

        # TODO: make configurable
        self.lpips_weight = 0.8
        self.face_weight = 0.1

    def accumulate_gradients(self, phase, full, fg, ibg, sync, gain):
        assert phase == 'rec'

        with torch.autograd.profiler.record_function('rec_forward'):
            gen_img, face_loss, lpips_loss = self.trainer.run_G(
                full,
                fg,
                ibg,
                sync=sync,
            )
            pixel_loss = F.mse_loss(gen_img, full)
            training_stats.report('loss/pixel', pixel_loss)
            training_stats.report('loss/face', face_loss)
            training_stats.report('loss/lpips', lpips_loss)
            loss = pixel_loss + \
                lpips_loss * self.lpips_weight + \
                face_loss * self.face_weight

        with torch.autograd.profiler.record_function('rec_backward'):
            loss.mul(gain).backward()


class SimpleRecLoss:
    def __init__(self, cfg, trainer, device):
        self.trainer = trainer
        self.device = device

    def accumulate_gradients(self, phase, real_img, sync, gain):
        assert phase == 'rec'

        with torch.autograd.profiler.record_function('rec_forward'):
            gen_img = self.trainer.run_G(real_img, sync=sync)
            loss = F.mse_loss(gen_img, real_img)
            training_stats.report('loss/rec', loss)

        with torch.autograd.profiler.record_function('rec_backward'):
            loss.mul(gain).backward()
