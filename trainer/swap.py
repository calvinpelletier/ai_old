#!/usr/bin/env python3
import torch
from ai_old.util.etc import print_
from ai_old.trainer.base import BaseTrainer
from external.sg2.misc import print_module_summary
from ai_old.util.factory import build_model
import external.sg2.misc as misc
from ai_old.trainer.phase import TrainingPhase
from external.optimizer import get_optimizer


class SynthOnlySwapTrainer(BaseTrainer):
    def run_batch(self, batch, batch_idx, cur_step):
        # prep_data
        with torch.autograd.profiler.record_function('data_prep'):
            phase_x_img = batch['x_img']
            phase_x_img = (
                phase_x_img.to(self.device).to(torch.float32) / 127.5 - 1
            ).split(self.batch_gpu)

            phase_x_gender = batch['x_gender']
            phase_x_gender = (
                phase_x_gender.to(self.device).to(torch.float32).unsqueeze(1)
            ).split(self.batch_gpu)

            phase_y_img = batch['y_img']
            phase_y_img = (
                phase_y_img.to(self.device).to(torch.float32) / 127.5 - 1
            ).split(self.batch_gpu)

            phase_y_gender = batch['y_gender']
            phase_y_gender = (
                phase_y_gender.to(self.device).to(torch.float32).unsqueeze(1)
            ).split(self.batch_gpu)

        # run training phases
        for phase in self.phases:
            if batch_idx % phase.interval != 0:
                continue

            phase.init_gradient_accumulation()

            # accumulate gradients over multiple rounds
            for round_idx, (x_img, x_gender, y_img, y_gender) in enumerate(zip(
                phase_x_img,
                phase_x_gender,
                phase_y_img,
                phase_y_gender,
            )):
                sync = (round_idx == self.batch_size // \
                    (self.batch_gpu * self.num_gpus) - 1)
                gain = phase.interval
                self.loss.accumulate_gradients(
                    phase=phase.name,
                    x_img=x_img,
                    x_gender=x_gender,
                    y_img=y_img,
                    y_gender=y_gender,
                    sync=sync,
                    gain=gain,
                )

            phase.update_params()


    def run_G(self, x_img, x_gender, y_img, y_gender, sync):
        G_e = self.ddp_modules['G_e']
        G_t = self.ddp_modules['G_t']

        # encode
        with misc.ddp_sync(G_e, sync):
            x_latent = G_e(x_img)
            y_latent = G_e(y_img)

        # transform
        with misc.ddp_sync(G_t, sync):
            x_latent_pred = G_t(y_latent, y_gender)
            y_latent_pred = G_t(x_latent, x_gender)

        return x_latent, y_latent, x_latent_pred, y_latent_pred


    def get_eval_fn(self):
        def _eval_fn(model, batch, batch_size, device):
            x_img = batch['x_img'].to(device).to(torch.float32) / 127.5 - 1
            y_img = batch['y_img'].to(device).to(torch.float32) / 127.5 - 1
            real_img = batch['real_img'].to(device).to(torch.float32) / 127.5 - 1
            x_gender = batch['x_gender'].to(device).to(
                torch.float32).unsqueeze(1)
            y_gender = batch['y_gender'].to(device).to(
                torch.float32).unsqueeze(1)
            real_gender = batch['real_gender'].to(device).to(
                torch.float32).unsqueeze(1)
            y_img_pred = model(x_img, x_gender)
            x_img_pred = model(y_img, y_gender)
            real_swap = model(real_img, real_gender)
            return x_img_pred, y_img_pred, real_swap
        return _eval_fn


    def get_sample_fn(self):
        def _sample_fn(model, batch, batch_size, device):
            x_img = batch['x_img'].to(device).to(torch.float32) / 127.5 - 1
            y_img = batch['y_img'].to(device).to(torch.float32) / 127.5 - 1
            real_img = batch['real_img'].to(device).to(torch.float32) / 127.5 - 1
            x_gender = batch['x_gender'].to(device).to(
                torch.float32).unsqueeze(1)
            y_gender = batch['y_gender'].to(device).to(
                torch.float32).unsqueeze(1)
            real_gender = batch['real_gender'].to(device).to(
                torch.float32).unsqueeze(1)
            y_img_pred = model(x_img, x_gender)
            x_img_pred = model(y_img, y_gender)
            real_swap = model(real_img, real_gender)
            x_img_pred = (
                x_img_pred * 127.5 + 128
            ).clamp(0, 255).to(torch.uint8)
            y_img_pred = (
                y_img_pred * 127.5 + 128
            ).clamp(0, 255).to(torch.uint8)
            real_swap = (
                real_swap * 127.5 + 128
            ).clamp(0, 255).to(torch.uint8)
            return x_img_pred, y_img_pred, real_swap
        return _sample_fn


    def _init_modules(self):
        cfg = self.cfg

        # build main model
        print_(self.rank, '[INFO] initializing model...')
        self.G = build_model(
            cfg,
            cfg.model,
        ).train().requires_grad_(False).to(self.device)

        # resume training
        assert not cfg.resume


    def _init_phases(self):
        print_(self.rank, '[INFO] initializing training phases...')
        self.phases = [self._build_loss_phase('swap', self.G)]


    def get_modules_for_save(self):
        return [('G', self.G)]


    def _get_modules_for_distribution(self):
        return [
            ('G_g', self.G.g, False),
            ('G_e', self.G.e, False),
            ('G_t', self.G.t, True),
        ]


    def print_models(self):
        img = torch.empty(
            [self.batch_gpu, 3, self.G.imsize, self.G.imsize],
            device=self.device,
        )
        gender = torch.empty([self.batch_gpu, 1], device=self.device)
        _ = print_module_summary(self.G, [img, gender])


class AdversarialSwapTrainer(BaseTrainer):
    def run_batch(self, batch, batch_idx, cur_step):
        # prep_data
        with torch.autograd.profiler.record_function('data_prep'):
            phase_real_img = batch['real_img']
            phase_real_img = (
                phase_real_img.to(self.device).to(torch.float32) / 127.5 - 1
            ).split(self.batch_gpu)

            phase_real_gender = batch['real_gender']
            phase_real_gender = (
                phase_real_gender.to(self.device).to(torch.float32).unsqueeze(1)
            ).split(self.batch_gpu)

            phase_x_img = batch['x_img']
            phase_x_img = (
                phase_x_img.to(self.device).to(torch.float32) / 127.5 - 1
            ).split(self.batch_gpu)

            phase_x_gender = batch['x_gender']
            phase_x_gender = (
                phase_x_gender.to(self.device).to(torch.float32).unsqueeze(1)
            ).split(self.batch_gpu)

            phase_y_img = batch['y_img']
            phase_y_img = (
                phase_y_img.to(self.device).to(torch.float32) / 127.5 - 1
            ).split(self.batch_gpu)

            phase_y_gender = batch['y_gender']
            phase_y_gender = (
                phase_y_gender.to(self.device).to(torch.float32).unsqueeze(1)
            ).split(self.batch_gpu)

        # run training phases
        for phase in self.phases:
            if batch_idx % phase.interval != 0:
                continue

            phase.init_gradient_accumulation()

            # accumulate gradients over multiple rounds
            for round_idx, (
                real_img,
                real_gender,
                x_img,
                x_gender,
                y_img,
                y_gender,
            ) in enumerate(zip(
                phase_real_img,
                phase_real_gender,
                phase_x_img,
                phase_x_gender,
                phase_y_img,
                phase_y_gender,
            )):
                sync = (round_idx == self.batch_size // \
                    (self.batch_gpu * self.num_gpus) - 1)
                gain = phase.interval
                self.loss.accumulate_gradients(
                    phase=phase.name,
                    real_img=real_img,
                    real_gender=real_gender,
                    x_img=x_img,
                    x_gender=x_gender,
                    y_img=y_img,
                    y_gender=y_gender,
                    sync=sync,
                    gain=gain,
                )

            phase.update_params()


    def run_G_real(self, real_img, real_gender, sync):
        G_e = self.ddp_modules['G_e']
        G_t = self.ddp_modules['G_t']

        # encode
        with misc.ddp_sync(G_e, sync):
            latent = G_e(real_img)

        # transform
        with misc.ddp_sync(G_t, sync):
            swap_latent = G_t(latent, real_gender)

        return swap_latent


    def run_G_synth(self, x_img, x_gender, y_img, y_gender, sync):
        G_e = self.ddp_modules['G_e']
        G_t = self.ddp_modules['G_t']

        # encode
        with misc.ddp_sync(G_e, sync):
            x_latent = G_e(x_img)
            y_latent = G_e(y_img)

        # transform
        with misc.ddp_sync(G_t, sync):
            x_latent_pred = G_t(y_latent, y_gender)
            y_latent_pred = G_t(x_latent, x_gender)

        return x_latent, y_latent, x_latent_pred, y_latent_pred


    def run_D_for_img(self, img, gender, sync):
        D_d = self.ddp_modules['D_d']
        D_e = self.ddp_modules['D_e']

        # encode
        with misc.ddp_sync(D_e, sync):
            latent = D_e(img)

        # discriminate
        with misc.ddp_sync(D_d, sync):
            logits = D_d(latent, gender)

        return logits


    def run_D_for_latent(self, latent, gender, sync):
        D_d = self.ddp_modules['D_d']

        # discriminate
        with misc.ddp_sync(D_d, sync):
            logits = D_d(latent, gender)

        return logits


    def get_eval_fn(self):
        def _eval_fn(model, batch, batch_size, device):
            x_img = batch['x_img'].to(device).to(torch.float32) / 127.5 - 1
            y_img = batch['y_img'].to(device).to(torch.float32) / 127.5 - 1
            real_img = batch['real_img'].to(device).to(torch.float32) / 127.5 - 1
            x_gender = batch['x_gender'].to(device).to(
                torch.float32).unsqueeze(1)
            y_gender = batch['y_gender'].to(device).to(
                torch.float32).unsqueeze(1)
            real_gender = batch['real_gender'].to(device).to(
                torch.float32).unsqueeze(1)
            y_img_pred = model(x_img, x_gender)
            x_img_pred = model(y_img, y_gender)
            real_swap = model(real_img, real_gender)
            return x_img_pred, y_img_pred, real_swap
        return _eval_fn


    def get_sample_fn(self):
        def _sample_fn(model, batch, batch_size, device):
            x_img = batch['x_img'].to(device).to(torch.float32) / 127.5 - 1
            y_img = batch['y_img'].to(device).to(torch.float32) / 127.5 - 1
            real_img = batch['real_img'].to(device).to(torch.float32) / 127.5 - 1
            x_gender = batch['x_gender'].to(device).to(
                torch.float32).unsqueeze(1)
            y_gender = batch['y_gender'].to(device).to(
                torch.float32).unsqueeze(1)
            real_gender = batch['real_gender'].to(device).to(
                torch.float32).unsqueeze(1)
            y_img_pred = model(x_img, x_gender)
            x_img_pred = model(y_img, y_gender)
            real_swap = model(real_img, real_gender)
            x_img_pred = (
                x_img_pred * 127.5 + 128
            ).clamp(0, 255).to(torch.uint8)
            y_img_pred = (
                y_img_pred * 127.5 + 128
            ).clamp(0, 255).to(torch.uint8)
            real_swap = (
                real_swap * 127.5 + 128
            ).clamp(0, 255).to(torch.uint8)
            return x_img_pred, y_img_pred, real_swap
        return _sample_fn


    def _init_modules(self):
        cfg = self.cfg

        # build models
        print_(self.rank, '[INFO] initializing model...')
        self.G = build_model(
            cfg,
            cfg.model.G,
        ).train().requires_grad_(False).to(self.device)
        self.D = build_model(
            cfg,
            cfg.model.D,
        ).train().requires_grad_(False).to(self.device)

        # resume training
        assert not cfg.resume


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
        return [('G', self.G), ('D', self.D)]


    def _get_modules_for_distribution(self):
        return [
            ('G_g', self.G.g, False),
            ('G_e', self.G.e, False),
            ('G_t', self.G.t, True),
            ('D_e', self.D.e, False),
            ('D_d', self.D.d, True),
        ]


    def print_models(self):
        img = torch.empty(
            [self.batch_gpu, 3, self.G.imsize, self.G.imsize],
            device=self.device,
        )
        gender = torch.empty([self.batch_gpu, 1], device=self.device)
        _ = print_module_summary(self.G, [img, gender])
        _ = print_module_summary(self.D, [img, gender])
