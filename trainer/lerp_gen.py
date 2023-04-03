#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from ai_old.util.etc import print_
from ai_old.trainer.base import BaseTrainer
from external.sg2.misc import print_module_summary
from ai_old.util.factory import build_model
import external.sg2.misc as misc
from ai_old.loss.perceptual.face import FaceIdLoss
from ai_old.loss.gender import GenderLoss
from ai_old.nn.models.lerp.dynamic import PretrainedGenerator


class LerpGenTrainer(BaseTrainer):
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


    def run_G(self, img, gender, w, sync):
        G = self.ddp_modules['G']
        img_generator = self.ddp_modules['img_generator']
        face_loss_model = self.ddp_modules['arcface']
        classify_loss_model = self.ddp_modules['classify']

        z1 = torch.randn([self.batch_size, G.z_dims], device=self.device)
        z2 = torch.randn([self.batch_size, G.z_dims], device=self.device)

        # generate lerps
        with misc.ddp_sync(G, sync):
            w1 = G(z1, w, gender, mag=1.)
            w2 = G(z2, w, gender, mag=1.)

        # generate imgs
        with misc.ddp_sync(img_generator, sync):
            img1 = img_generator(w1)
            img2 = img_generator(w2)

        # delta loss
        delta_loss = F.mse_loss(w1, w) + F.mse_loss(w2, w)

        # face id loss
        with misc.ddp_sync(face_loss_model, sync):
            face_loss = face_loss_model(img1, img).mean() + \
                face_loss_model(img2, img).mean()

        # classification loss
        with misc.ddp_sync(classify_loss_model, sync):
            classify_loss = classify_loss_model(img1, 1. - gender).mean() + \
                classify_loss_model(img2, 1. - gender).mean()

        # regularization
        reg_loss = F.mse_loss(
            (w2.mean(dim=1) - w1.mean(dim=1)).norm(dim=1) * self.reg_scale,
            (z2 - z1).norm(dim=1),
        )

        return delta_loss, face_loss, classify_loss, reg_loss


    def get_eval_fn(self):
        def _eval_fn(model, batch, batch_size, device):
            img = batch['img'].to(device).to(torch.float32) / 127.5 - 1
            gender = batch['gender'].to(device).to(
                torch.float32).unsqueeze(1)
            w = batch['w'].to(device).to(torch.float32)
            z1 = torch.randn([batch_size, model.z_dims], device=device)
            z2 = torch.randn([batch_size, model.z_dims], device=device)

            w1 = model(z1, w, gender, mag=1.)
            w2 = model(z2, w, gender, mag=1.)

            img1 = self.img_generator(w1)
            img2 = self.img_generator(w2)

            delta_loss = F.mse_loss(w1, w) + F.mse_loss(w2, w)

            face_loss = self.face_loss_model(img1, img).mean() + \
                self.face_loss_model(img2, img).mean()

            c_loss1 = self.classify_loss_model(img1, 1. - gender).mean()
            c_loss2 = self.classify_loss_model(img2, 1. - gender).mean()
            classify_loss = c_loss1 + c_loss2

            reg_loss = F.mse_loss(
                (w2 - w1).norm(dim=1) * self.reg_scale,
                (z2 - z1).norm(dim=1),
            )

            return delta_loss, face_loss, classify_loss, reg_loss
        return _eval_fn


    def get_sample_fn(self):
        def _sample_fn(model, batch, batch_size, device):
            gender = batch['gender'].to(device).to(
                torch.float32).unsqueeze(1)
            w = batch['w'].to(device).to(torch.float32)
            samples = []
            for i in range(4):
                z = torch.randn([batch_size, model.z_dims], device=device)
                new_w = model(z, w, gender, mag=1.)
                new_img = self.img_generator(new_w)
                samples.append(new_img)
            return samples
        return _sample_fn


    def get_model_key_for_eval(self):
        return 'G'


    def _init_modules(self):
        cfg = self.cfg

        # build main model
        print_(self.rank, '[INFO] initializing model...')
        self.G = build_model(
            cfg,
            cfg.model,
        ).train().requires_grad_(False).to(self.device)

        # img generator
        self.img_generator = PretrainedGenerator(
            cfg.dataset.imsize).eval().requires_grad_(False).to(self.device)

        # face loss
        self.face_loss_model = FaceIdLoss(
            cfg.dataset.imsize).eval().requires_grad_(False).to(self.device)

        # classification loss
        self.classify_loss_model = GenderLoss(
            l2=cfg.loss.classify.l2,
        ).eval().requires_grad_(False).to(self.device)

        # regularization
        self.reg_scale = cfg.loss.reg.scale

        # resume training
        assert not cfg.resume


    def _init_phases(self):
        print_(self.rank, '[INFO] initializing training phases...')
        self.phases = [self._build_loss_phase('main', self.G)]


    def get_modules_for_save(self):
        return [('img_generator', self.img_generator)]


    def _get_modules_for_distribution(self):
        return [
            ('G', self.G, True),
            ('img_generator', self.img_generator, False),
            ('arcface', self.face_loss_model, False),
            ('classify', self.classify_loss_model, False),
        ]


    def print_models(self):
        w_plus = torch.empty([self.batch_gpu, 18, 512], device=self.device)
        gender = torch.empty([self.batch_gpu, 1], device=self.device)
        z = torch.empty([self.batch_gpu, self.G.z_dims], device=self.device)
        _ = print_module_summary(self.G, [z, w_plus, gender])
