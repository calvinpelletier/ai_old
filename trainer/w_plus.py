#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from ai_old.util.etc import print_
from ai_old.trainer.base import BaseTrainer
from external.sg2.misc import print_module_summary
from ai_old.util.factory import build_model
import external.sg2.misc as misc
from ai_old.loss.perceptual.face import FaceIdLoss
from ai_old.loss.clip import GenderSwapClipLoss, GenderSwapClipDirLoss


class RealOnlySwapTrainer(BaseTrainer):
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

            phase_real_w = batch['real_w']
            phase_real_w = (
                phase_real_w.to(self.device).to(torch.float32)
            ).split(self.batch_gpu)

        # run training phases
        for phase in self.phases:
            if batch_idx % phase.interval != 0:
                continue

            phase.init_gradient_accumulation()

            # accumulate gradients over multiple rounds
            for round_idx, (real_img, real_gender, real_w) in enumerate(zip(
                phase_real_img,
                phase_real_gender,
                phase_real_w,
            )):
                sync = (round_idx == self.batch_size // \
                    (self.batch_gpu * self.num_gpus) - 1)
                gain = phase.interval
                self.loss.accumulate_gradients(
                    phase=phase.name,
                    real_img=real_img,
                    real_gender=real_gender,
                    real_w=real_w,
                    sync=sync,
                    gain=gain,
                )

            phase.update_params()


    def run_G(self, real_img, real_gender, real_w, sync):
        G_g = self.ddp_modules['G_g']
        G_f = self.ddp_modules['G_f']
        face_loss_model = self.ddp_modules['arcface']
        clip_loss_model = self.ddp_modules['clip']

        # swap latent
        with misc.ddp_sync(G_f, sync):
            swap_w, delta = G_f(real_w, real_gender, magnitude=1.)

        # generate swap img
        with misc.ddp_sync(G_g, sync):
            swap_img = G_g(swap_w)

        # face id loss
        with misc.ddp_sync(face_loss_model, sync):
            face_loss = face_loss_model(swap_img, real_img)

        # clip loss
        with misc.ddp_sync(clip_loss_model, sync):
            if self.cfg.loss.clip.dir:
                clip_loss = clip_loss_model(real_img, swap_img, real_gender)
            else:
                clip_loss = clip_loss_model(swap_img, real_gender)

        return clip_loss, face_loss, delta



    def get_eval_fn(self):
        def _eval_fn(G, batch, batch_size, device):
            real_img = batch['real_img'].to(device).to(torch.float32) / 127.5 - 1
            real_gender = batch['real_gender'].to(device).to(
                torch.float32).unsqueeze(1)
            real_w = batch['real_w'].to(device).to(torch.float32)
            swap_w, delta = G.f(real_w, real_gender, magnitude=1.)
            swap_img = G.g(swap_w)
            return real_img, real_gender, swap_img, delta
        return _eval_fn


    def get_sample_fn(self):
        def _sample_fn(model, batch, batch_size, device, sample_imsize):
            real_gender = batch['real_gender'].to(device).to(
                torch.float32).unsqueeze(1)
            real_w = batch['real_w'].to(device).to(torch.float32)

            swaps = []
            mags = [0.5, 1., 1.5]
            for mag in mags:
                swap = model(real_w, real_gender, magnitude=mag)
                swap = F.interpolate(
                    swap,
                    size=(sample_imsize, sample_imsize),
                    mode='bilinear',
                    align_corners=True,
                )
                swap = (swap * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                swaps.append(swap)
            return swaps
        return _sample_fn


    def _init_modules(self):
        cfg = self.cfg

        # build main model
        print_(self.rank, '[INFO] initializing model...')
        self.G = build_model(
            cfg,
            cfg.model,
        ).train().requires_grad_(False).to(self.device)

        # build loss models
        self.face_loss_model = FaceIdLoss(
            cfg.dataset.imsize).eval().requires_grad_(False).to(self.device)

        if cfg.loss.clip.dir:
            clip_loss_module = GenderSwapClipDirLoss
        else:
            clip_loss_module = GenderSwapClipLoss
        self.clip_loss_model = clip_loss_module(
            cfg.loss.clip.female_male_target_texts,
            self.device,
        ).eval().requires_grad_(False).to(self.device)

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
            ('G_f', self.G.f, True),
            ('arcface', self.face_loss_model, False),
            ('clip', self.clip_loss_model, False),
        ]


    def print_models(self):
        w_plus = torch.empty([self.batch_gpu, 18, 512], device=self.device)
        gender = torch.empty([self.batch_gpu, 1], device=self.device)
        _ = print_module_summary(self.G.f, [w_plus, gender])


class NullSwapTrainer(RealOnlySwapTrainer):
    def run_batch(self, batch, batch_idx, cur_step):
        pass


    def run_G(self, real_img, real_gender, real_w, sync):
        pass


    def _init_modules(self):
        cfg = self.cfg

        # build main model
        print_(self.rank, '[INFO] initializing model...')
        self.G = build_model(
            cfg,
            cfg.model,
        ).train().requires_grad_(False).to(self.device)

        # build loss models (for task only)
        self.face_loss_model = FaceIdLoss(
            cfg.dataset.imsize).eval().requires_grad_(False).to(self.device)

        self.clip_loss_model = GenderSwapClipDirLoss(
            ['male face', 'female face'],
            self.device,
        ).eval().requires_grad_(False).to(self.device)

        # resume training
        assert not cfg.resume


    def _init_loss(self):
        pass


    def _init_phases(self):
        self.phases = []


    def get_modules_for_save(self):
        return [('G', self.G)]


    def _get_modules_for_distribution(self):
        return [
            ('G_g', self.G.g, False),
            ('G_f', self.G.f, False),
            ('arcface', self.face_loss_model, False),
            ('clip', self.clip_loss_model, False),
        ]


    def print_models(self):
        w_plus = torch.empty([self.batch_gpu, 18, 512], device=self.device)
        gender = torch.empty([self.batch_gpu, 1], device=self.device)
        _ = print_module_summary(self.G.f, [w_plus, gender])
