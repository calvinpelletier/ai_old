#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from ai_old.util.etc import print_
from ai_old.trainer.aae import AaeTrainer
from external.sg2.misc import print_module_summary
from ai_old.util.factory import build_model, build_augpipe
import copy
from external.sg2 import training_stats
import external.sg2.misc as misc
from ai_old.util.factory import build_model_from_exp
from ai_old.loss.perceptual.face import FaceIdLoss
from ai_old.loss.perceptual.trad import PerceptualLoss
from external.optimizer import get_optimizer
from ai_old.trainer.phase import TrainingPhase
import lpips
from ai_old.loss.perceptual.ahanu import AhanuPercepLoss


class AsaeTrainer(AaeTrainer):
    def run_batch(self, batch, batch_idx, cur_step):
        # prep_data
        with torch.autograd.profiler.record_function('data_prep'):
            phase_real = batch['img']
            phase_real = (
                phase_real.to(self.device).to(torch.float32) / 127.5 - 1
            ).split(self.batch_gpu)

            phase_fg_mask = batch['mask'].unsqueeze(1)
            phase_fg_mask = (
                phase_fg_mask.to(self.device).to(torch.float32) / 255.
            ).split(self.batch_gpu)

        # run training phases
        for phase in self.phases:
            if batch_idx % phase.interval != 0:
                continue

            phase.init_gradient_accumulation()

            # accumulate gradients over multiple rounds
            for round_idx, (real, fg_mask) in enumerate(zip(
                phase_real,
                phase_fg_mask,
            )):
                sync = (round_idx == self.batch_size // \
                    (self.batch_gpu * self.num_gpus) - 1)
                gain = phase.interval
                self.loss.accumulate_gradients(
                    phase=phase.name,
                    real=real,
                    fg_mask=fg_mask,
                    sync=sync,
                    gain=gain,
                )

            phase.update_params()

        self._update_model_ema(cur_step)

        self._run_ada_heuristic(batch_idx)


    def run_G(self, real, fg_mask, do_rec_losses, sync):
        G_g = self.ddp_modules['G_g']
        G_e = self.ddp_modules['G_e']
        face_loss_model = self.ddp_modules['arcface']
        perceptual_loss_model = self.ddp_modules['perceptual']

        # encode
        with misc.ddp_sync(G_e, sync):
            intermediate = G_e(real)

        # synthesize
        with misc.ddp_sync(G_g, sync):
            output = G_g(intermediate)

        if not do_rec_losses:
            return output

        # mask
        output_masked = output * fg_mask
        real_masked = real * fg_mask

        # face loss
        with misc.ddp_sync(face_loss_model, sync):
            face_loss = face_loss_model(output_masked, real_masked.detach())

        # perceptual loss
        with misc.ddp_sync(perceptual_loss_model, sync):
            perceptual_loss = perceptual_loss_model(
                output_masked, real_masked.detach())

        pixel_loss = F.mse_loss(output_masked, real_masked.detach())

        return pixel_loss, face_loss, perceptual_loss


    def get_eval_fn(self):
        def eval_fn(model, batch, batch_size, device):
            real = batch['img'].to(device).to(torch.float32) / 127.5 - 1
            fg_mask = batch['mask'].unsqueeze(1).to(self.device).to(
                torch.float32) / 255.
            rec = model(real)
            real_masked = real * fg_mask
            rec_masked = rec * fg_mask
            return real_masked, rec_masked
        return eval_fn


    def get_sample_fn(self):
        def sample_fn(model, batch, batch_size, device):
            real = batch['img'].to(device).to(torch.float32) / 127.5 - 1
            rec = model(real)
            rec = (rec * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return rec
        return sample_fn
