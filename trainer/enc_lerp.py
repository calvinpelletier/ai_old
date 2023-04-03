#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from ai_old.util.etc import print_
from ai_old.trainer.base import BaseTrainer
from external.sg2.misc import print_module_summary
from ai_old.util.factory import build_model, build_model_from_exp
import external.sg2.misc as misc
from random import random
from torch.distributions.uniform import Uniform
import numpy as np


MAX_MAG = 1.5
N_MAGS = 8


class EncLerpTrainer(BaseTrainer):
    def run_batch(self, batch, batch_idx, cur_step):
        # prep_data
        with torch.autograd.profiler.record_function('data_prep'):
            phase_img = batch['img']
            phase_img = (
                phase_img.to(self.device).to(torch.float32) / 127.5 - 1
            ).split(self.batch_gpu)

            phase_gender = batch['gender']
            phase_gender = (
                phase_gender.to(self.device).to(torch.float32).unsqueeze(1)
            ).split(self.batch_gpu)

            phase_w = batch['w']
            phase_w = (
                phase_w.to(self.device).to(torch.float32)
            ).split(self.batch_gpu)

        # run training phases
        for phase in self.phases:
            if batch_idx % phase.interval != 0:
                continue

            phase.init_gradient_accumulation()

            # accumulate gradients over multiple rounds
            for round_idx, (img, gender, w) in enumerate(zip(
                phase_img,
                phase_gender,
                phase_w,
            )):
                sync = (round_idx == self.batch_size // \
                    (self.batch_gpu * self.num_gpus) - 1)
                gain = phase.interval
                self.loss.accumulate_gradients(
                    phase=phase.name,
                    img=img,
                    gender=gender,
                    w=w,
                    sync=sync,
                    gain=gain,
                )

            phase.update_params()


    def run_model(self, img, gender, w, sync):
        w_lerper = self.ddp_modules['w_lerper']
        img_generator = self.ddp_modules['img_generator']
        encoder = self.ddp_modules['encoder']
        enc_lerper = self.ddp_modules['enc_lerper']

        bs = img.shape[0]
        mag = self.mag_sampler.sample((bs,)).to(self.device)

        with torch.no_grad():
            # w lerp
            with misc.ddp_sync(w_lerper, sync):
                guide_w = w_lerper(
                    w,
                    gender,
                    magnitude=MAX_MAG,
                )
                target_w = w_lerper(w, gender, magnitude=mag)

            # generate
            with misc.ddp_sync(img_generator, sync):
                guide_img = img_generator(guide_w)
                target_img = img_generator(target_w)

            # encode
            with misc.ddp_sync(encoder, sync):
                enc = encoder(img)
                guide_enc = encoder(guide_img)
                target_enc = encoder(target_img)

        # enc lerp
        with misc.ddp_sync(enc_lerper, sync):
            pred_enc = enc_lerper(enc.detach(), guide_enc.detach(), mag.detach())

        # print(pred_enc.shape, target_enc.shape)
        # print(torch.mean(torch.abs(pred_enc.detach() - target_enc.detach())))

        return pred_enc, target_enc


    def get_eval_fn(self):
        def _eval_fn(model, batch, batch_size, device):
            img = batch['img'].to(device).to(torch.float32) / 127.5 - 1
            gender = batch['gender'].to(device).to(
                torch.float32).unsqueeze(1)
            w = batch['w'].to(device).to(torch.float32)

            mag = self.mag_sampler.sample((batch_size,)).to(device)

            # w lerp
            guide_w = self.lerp_and_gen.f(
                w,
                gender,
                magnitude=MAX_MAG,
            )
            target_w = self.lerp_and_gen.f(w, gender, magnitude=mag)

            # generate
            guide_img = self.lerp_and_gen.g(guide_w)
            target_img = self.lerp_and_gen.g(target_w)

            # encode
            enc = self.ae.e(img)
            guide_enc = self.ae.e(guide_img)
            target_enc = self.ae.e(target_img)

            # enc lerp
            pred_enc = model(enc, guide_enc, mag)
            return pred_enc, target_enc
        return _eval_fn


    def get_sample_fn(self):
        def _sample_fn(model, batch, batch_size, device):
            img = batch['img'].to(device).to(torch.float32) / 127.5 - 1
            gender = batch['gender'].to(device).to(
                torch.float32).unsqueeze(1)
            w = batch['w'].to(device).to(torch.float32)

            # w lerp
            guide_w = self.lerp_and_gen.f(w, gender, magnitude=MAX_MAG)

            # w to img
            guide_img = self.lerp_and_gen.g(guide_w)

            # encode
            enc = self.ae.e(img)
            guide_enc = self.ae.e(guide_img)

            # enc lerp
            pred_enc0 = model(
                enc,
                guide_enc,
                torch.tensor((0.,), device=device).repeat(batch_size),
            )
            pred_enc1 = model(
                enc,
                guide_enc,
                torch.tensor((0.5,), device=device).repeat(batch_size),
            )
            pred_enc2 = model(
                enc,
                guide_enc,
                torch.tensor((1.,), device=device).repeat(batch_size),
            )
            pred_enc3 = model(
                enc,
                guide_enc,
                torch.tensor((1.5,), device=device).repeat(batch_size),
            )

            # enc to img
            img0 = self.ae.g(pred_enc0)
            img1 = self.ae.g(pred_enc1)
            img2 = self.ae.g(pred_enc2)
            img3 = self.ae.g(pred_enc3)
            # img0 = self.ae.g(enc)
            # img1 = self.ae.g(guide_enc)
            # img2 = self.ae.g(guide_enc)
            # img3 = self.ae.g(guide_enc)
            return img0, img1, img2, img3
        return _sample_fn


    def get_model_key_for_eval(self):
        return 'model'


    def _init_modules(self):
        cfg = self.cfg

        # main model
        print_(self.rank, '[INFO] initializing model...')
        self.model = build_model(
            cfg,
            cfg.model,
        ).train().requires_grad_(False).to(self.device)

        # w lerp and gen
        self.lerp_and_gen = build_model_from_exp(
            cfg.trainer.w_lerp_exp,
            'G',
            return_cfg=False,
        ).eval().requires_grad_(False).to(self.device)

        # autoencode
        self.ae = build_model_from_exp(
            cfg.trainer.ae_exp,
            'G_ema',
            return_cfg=False,
        ).eval().requires_grad_(False).to(self.device)

        self.mag_sampler = Uniform(0., MAX_MAG)

        # resume training
        assert not cfg.resume


    def _init_phases(self):
        print_(self.rank, '[INFO] initializing training phases...')
        self.phases = [self._build_loss_phase('main', self.model)]


    def get_modules_for_save(self):
        return [('model', self.model)]


    def _get_modules_for_distribution(self):
        return [
            ('w_lerper', self.lerp_and_gen.f, False),
            ('img_generator', self.lerp_and_gen.g, False),
            ('encoder', self.ae.e, False),
            ('enc_lerper', self.model, True),
        ]


    def print_models(self):
        enc = torch.empty([self.batch_gpu, 512, 4, 4], device=self.device)
        mag = torch.ones([self.batch_gpu,], device=self.device)
        _ = print_module_summary(self.model, [enc, enc, mag])


class FastEncLerpTrainer(EncLerpTrainer):
    def run_batch(self, batch, batch_idx, cur_step):
        # prep_data
        with torch.autograd.profiler.record_function('data_prep'):
            phase_target_enc = batch['target_enc']
            phase_target_enc = (
                phase_target_enc.to(self.device).to(torch.float32)
            ).split(self.batch_gpu)

        # run training phases
        for phase in self.phases:
            if batch_idx % phase.interval != 0:
                continue

            phase.init_gradient_accumulation()

            # accumulate gradients over multiple rounds
            for round_idx, target_enc in enumerate(phase_target_enc):
                sync = (round_idx == self.batch_size // \
                    (self.batch_gpu * self.num_gpus) - 1)
                gain = phase.interval
                self.loss.accumulate_gradients(
                    phase=phase.name,
                    target_enc=target_enc,
                    sync=sync,
                    gain=gain,
                )

            phase.update_params()


    def run_model(self, target_enc, sync):
        # target enc shape: (b, N_MAGS, c, h, w)
        base_enc = target_enc[:, 0, :, :, :]
        guide_enc = target_enc[:, N_MAGS - 1, :, :, :]
        model = self.ddp_modules['model']
        pred_enc = torch.zeros_like(target_enc)
        with misc.ddp_sync(model, sync):
            for i in range(N_MAGS):
                pred_enc[:, i, :, :, :] = model(
                    base_enc.detach(),
                    guide_enc.detach(),
                    self.mags[i].detach(),
                )
        return pred_enc


    def get_eval_fn(self):
        def _eval_fn(model, batch, batch_size, device):
            target_enc = batch['target_enc'].to(device).to(torch.float32)
            base_enc = target_enc[:, 0, :, :, :]
            guide_enc = target_enc[:, N_MAGS - 1, :, :, :]
            pred_enc = torch.zeros_like(target_enc)
            for i in range(N_MAGS):
                pred_enc[:, i, :, :, :] = model(
                    base_enc,
                    guide_enc,
                    self.mags[i],
                )
            return pred_enc, target_enc
        return _eval_fn


    def get_sample_fn(self):
        def _sample_fn(model, batch, batch_size, device):
            target_enc = batch['target_enc'].to(device).to(torch.float32)
            base_enc = target_enc[:, 0, :, :, :]
            guide_enc = target_enc[:, N_MAGS - 1, :, :, :]

            # enc lerp
            pred_enc0 = model(
                base_enc,
                guide_enc,
                torch.tensor((0.,), device=device).repeat(batch_size),
            )
            pred_enc1 = model(
                base_enc,
                guide_enc,
                torch.tensor((0.5,), device=device).repeat(batch_size),
            )
            pred_enc2 = model(
                base_enc,
                guide_enc,
                torch.tensor((1.,), device=device).repeat(batch_size),
            )
            pred_enc3 = model(
                base_enc,
                guide_enc,
                torch.tensor((1.5,), device=device).repeat(batch_size),
            )

            # enc to img
            img0 = self.ae_g(pred_enc0)
            img1 = self.ae_g(pred_enc1)
            img2 = self.ae_g(pred_enc2)
            img3 = self.ae_g(pred_enc3)
            return img0, img1, img2, img3
        return _sample_fn


    def _init_modules(self):
        cfg = self.cfg

        # main model
        print_(self.rank, '[INFO] initializing model...')
        self.model = build_model(
            cfg,
            cfg.model,
        ).train().requires_grad_(False).to(self.device)

        # autoencode
        self.ae_g = build_model_from_exp(
            cfg.trainer.ae_exp,
            'G_ema',
            return_cfg=False,
        ).g.eval().requires_grad_(False).to(self.device)

        # mags
        self.mags = []
        for mag in np.linspace(0., MAX_MAG, num=N_MAGS):
            self.mags.append(torch.tensor(
                (mag,),
                device=self.device,
                dtype=torch.float32,
            ).repeat(cfg.dataset.batch_size))

        # resume training
        assert not cfg.resume


    def _get_modules_for_distribution(self):
        return [
            ('model', self.model, True),
        ]
