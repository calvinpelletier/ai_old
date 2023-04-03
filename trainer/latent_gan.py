#!/usr/bin/env python3
import torch
from ai_old.util.etc import print_
from ai_old.trainer.base import BaseTrainer
from external.sg2.misc import print_module_summary
from ai_old.util.factory import build_model, build_augpipe
import copy
from external.sg2 import training_stats
import external.sg2.misc as misc
import numpy as np
import external.sg2.legacy as legacy
import os
from ai_old.trainer.phase import TrainingPhase
from external.optimizer import get_optimizer


class LatentGanTrainer(BaseTrainer):
    def run_batch(self, batch, batch_idx, cur_step):
        # prep_data
        with torch.autograd.profiler.record_function('data_prep'):
            phase_real_img = batch['y']
            phase_real_img = (
                phase_real_img.to(self.device).to(torch.float32) / 127.5 - 1
            ).split(self.batch_gpu)
            all_z = torch.randn(
                [len(self.phases) * self.batch_size, self.G.z_dims],
                device=self.device,
            )
            all_z = [
                phase_z.split(self.batch_gpu) \
                for phase_z in all_z.split(self.batch_size)
            ]

        # run training phases
        for phase, phase_z in zip(self.phases, all_z):
            if batch_idx % phase.interval != 0:
                continue

            phase.init_gradient_accumulation()

            # accumulate gradients over multiple rounds
            for round_idx, (real_img, z) in enumerate(zip(
                phase_real_img,
                phase_z,
            )):
                sync = (round_idx == self.batch_size // \
                    (self.batch_gpu * self.num_gpus) - 1)
                gain = phase.interval
                self.loss.accumulate_gradients(
                    phase=phase.name,
                    real=real_img,
                    seed=z,
                    sync=sync,
                    gain=gain,
                )

            phase.update_params()

        self._update_model_ema(cur_step)


    def run_G(self, seed, sync):
        G_f = self.ddp_modules['G_f']

        # seed to latent
        with misc.ddp_sync(G_f, sync):
            latent = G_f(seed)

        return latent


    def run_D_for_img(self, img, sync):
        D_d = self.ddp_modules['D_d']
        D_e = self.ddp_modules['D_e']

        # encode
        with misc.ddp_sync(D_e, sync):
            latent = D_e(img)

        # discriminate
        with misc.ddp_sync(D_d, sync):
            logits = D_d(latent)

        return logits


    def run_D_for_latent(self, latent, sync):
        D_d = self.ddp_modules['D_d']

        # discriminate
        with misc.ddp_sync(D_d, sync):
            logits = D_d(latent)

        return logits


    def get_eval_fn(self):
        def eval_fn(model, _batch, batch_size, device):
            seed = torch.randn([batch_size, model.z_dims], device=device)
            img = model(seed)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return img
        return eval_fn


    def get_sample_fn(self):
        def sample_fn(model, _batch, batch_size, device):
            seed = torch.randn([batch_size, model.z_dims], device=device)
            img = model(seed)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return img
        return sample_fn


    def _update_model_ema(self, cur_step):
        with torch.autograd.profiler.record_function('Gema'):
            ema_steps = self.ema_ksteps * 1000
            if self.ema_rampup is not None:
                ema_steps = min(ema_steps, cur_step * self.ema_rampup)
            ema_beta = 0.5 ** (self.batch_size / max(ema_steps, 1e-8))
            for p_ema, p in zip(self.G_ema.parameters(), self.G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(self.G_ema.buffers(), self.G.buffers()):
                b_ema.copy_(b)


    def _init_modules(self):
        cfg = self.cfg

        # TODO: initialized aae should be considered not from scratch
        from_scratch = not cfg.resume

        # build models
        print_(self.rank, '[INFO] initializing models...')
        self.G = build_model(
            cfg,
            cfg.model.G,
        ).train().requires_grad_(False).to(self.device)
        self.D = build_model(
            cfg,
            cfg.model.D,
        ).train().requires_grad_(False).to(self.device)

        # model ema
        self.G_ema = copy.deepcopy(self.G).eval()
        self.ema_ksteps = 10
        self.ema_rampup = 0.05 if from_scratch else None

        # resume training
        assert not cfg.resume, 'todo'


    def _init_phases(self):
        print_(self.rank, '[INFO] initializing training phases...')
        cfg = self.cfg
        self.phases = []

        # generator phase
        self.phases.append(TrainingPhase(
            name='G',
            module=self.G,
            opt=get_optimizer(
                cfg,
                cfg.opt.G,
                self.G.parameters(),
                reg_interval=None,
            ),
            interval=1,
            device=self.device,
            rank=self.rank,
        ))

        # discriminator phases
        self.phases += self._build_regularized_loss_phases(
            'D',
            self.D,
            cfg.loss.D.gp.freq,
        )


    def get_modules_for_save(self):
        return [
            ('G', self.G),
            ('D', self.D),
            ('G_ema', self.G_ema),
        ]


    def _get_modules_for_distribution(self):
        return [
            ('G_f', self.G.f, True),
            ('G_g', self.G.g, False),
            ('D_e', self.D.e, False),
            ('D_d', self.D.d, True),
            (None, self.G_ema, False),
        ]


    def print_models(self):
        seed = torch.empty(
            [self.batch_gpu, self.G.z_dims],
            device=self.device,
        )
        _ = print_module_summary(self.G, [seed])

        img = torch.empty(
            [self.batch_gpu, 3, self.G.imsize, self.G.imsize],
            device=self.device,
        )
        print_module_summary(self.D, [img])
