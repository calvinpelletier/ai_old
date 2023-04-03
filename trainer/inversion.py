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





class GanInverter(BaseTrainer):
    def run_batch(self, _batch, batch_idx, cur_step):
        # prep_data
        with torch.autograd.profiler.record_function('data_prep'):
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
            for round_idx, z in enumerate(phase_z):
                sync = (round_idx == self.batch_size // \
                    (self.batch_gpu * self.num_gpus) - 1)
                gain = phase.interval
                self.loss.accumulate_gradients(
                    phase=phase.name,
                    gen_z=z,
                    sync=sync,
                    gain=gain,
                )

            phase.update_params()


    def run_G(self, z, sync):
        G_f = self.ddp_modules['G_f']
        G_g = self.ddp_modules['G_g']
        G_e = self.ddp_modules['G_e']

        # disentangle latent vector
        with misc.ddp_sync(G_f, sync):
            ws = G_f(z, None)

        # synthesize image
        with misc.ddp_sync(G_g, sync):
            img = G_g(ws)

        # encode
        with misc.ddp_sync(G_e, sync):
            enc = G_e(img)

        return ws[:, 0, :], enc


    def get_eval_fn(self):
        def _eval_fn(model, batch, batch_size, device):
            z = batch['z'].to(device)
            gen_img, rec_img = model(z=z)
            return gen_img, rec_img
        return _eval_fn


    def get_sample_fn(self):
        def _sample_fn(model, batch, batch_size, device):
            z = batch['z'].to(device)
            # z = torch.randn([batch_size, model.z_dims], device=device)
            gen_img, rec_img = model(z=z)
            gen_img = (gen_img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            rec_img = (rec_img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return gen_img, rec_img
        return _sample_fn


    def _init_modules(self):
        cfg = self.cfg

        # build models
        print_(self.rank, '[INFO] initializing model...')
        self.G = build_model(
            cfg,
            cfg.model,
        ).train().requires_grad_(False).to(self.device)

        # resume training
        assert not cfg.resume


    def _init_phases(self):
        print_(self.rank, '[INFO] initializing training phases...')
        self.phases = [self._build_loss_phase('enc', self.G)]


    def get_modules_for_save(self):
        return [('G', self.G)]


    def _get_modules_for_distribution(self):
        return [
            ('G_f', self.G.f, False),
            ('G_g', self.G.g, False),
            ('G_e', self.G.e, True),
        ]


    def print_models(self):
        z = torch.empty(
            [self.batch_gpu, self.G.z_dims],
            device=self.device,
        )
        gen_img, rec_img = print_module_summary(self.G, [z])
