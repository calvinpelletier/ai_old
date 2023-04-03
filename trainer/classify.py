#!/usr/bin/env python3
import torch
from ai_old.util.etc import print_
from ai_old.trainer.base import BaseTrainer
from external.sg2.misc import print_module_summary
from ai_old.util.factory import build_model
import external.sg2.misc as misc


class FullAttrClassificationTrainer(BaseTrainer):
    def run_batch(self, batch, batch_idx, cur_step):
        # prep_data
        with torch.autograd.profiler.record_function('data_prep'):
            phase_img = batch['img']
            phase_img = (
                phase_img.to(self.device).to(torch.float32) / 127.5 - 1
            ).split(self.batch_gpu)

            phase_gender = batch['gender']
            phase_gender = (
                phase_gender.to(self.device).to(torch.float32)
            ).split(self.batch_gpu)

            phase_age = batch['age']
            phase_age = (
                phase_age.to(self.device).to(torch.float32)
            ).split(self.batch_gpu)

            phase_glasses = batch['glasses']
            phase_glasses = (
                phase_glasses.to(self.device).to(torch.float32)
            ).split(self.batch_gpu)

            phase_smile = batch['smile']
            phase_smile = (
                phase_smile.to(self.device).to(torch.float32)
            ).split(self.batch_gpu)

            phase_mouth = batch['mouth']
            phase_mouth = (
                phase_mouth.to(self.device).to(torch.float32)
            ).split(self.batch_gpu)

            phase_hair_len = batch['hair_len']
            phase_hair_len = (
                phase_hair_len.to(self.device).to(torch.float32)
            ).split(self.batch_gpu)

            phase_hair_curl = batch['hair_curl']
            phase_hair_curl = (
                phase_hair_curl.to(self.device).to(torch.float32)
            ).split(self.batch_gpu)

            phase_baldness = batch['baldness']
            phase_baldness = (
                phase_baldness.to(self.device).to(torch.float32)
            ).split(self.batch_gpu)

            phase_facial_hair = batch['facial_hair']
            phase_facial_hair = (
                phase_facial_hair.to(self.device).to(torch.float32)
            ).split(self.batch_gpu)

            phase_makeup = batch['makeup']
            phase_makeup = (
                phase_makeup.to(self.device).to(torch.float32)
            ).split(self.batch_gpu)


        # run training phases
        for phase in self.phases:
            if batch_idx % phase.interval != 0:
                continue

            phase.init_gradient_accumulation()

            # accumulate gradients over multiple rounds
            for round_idx, (
                img,
                gender,
                age,
                glasses,
                smile,
                mouth,
                hair_len,
                hair_curl,
                baldness,
                facial_hair,
                makeup,
            ) in enumerate(zip(
                phase_img,
                phase_gender,
                phase_age,
                phase_glasses,
                phase_smile,
                phase_mouth,
                phase_hair_len,
                phase_hair_curl,
                phase_baldness,
                phase_facial_hair,
                phase_makeup,
            )):
                sync = (round_idx == self.batch_size // \
                    (self.batch_gpu * self.num_gpus) - 1)
                gain = phase.interval
                self.loss.accumulate_gradients(
                    phase=phase.name,
                    img=img,
                    gender=gender,
                    age=age,
                    glasses=glasses,
                    smile=smile,
                    mouth=mouth,
                    hair_len=hair_len,
                    hair_curl=hair_curl,
                    baldness=baldness,
                    facial_hair=facial_hair,
                    makeup=makeup,
                    sync=sync,
                    gain=gain,
                )

            phase.update_params()


    def run_model(self, img, sync):
        model = self.ddp_modules['model']
        with misc.ddp_sync(model, sync):
            preds = model(img)
        return preds


    def get_eval_fn(self):
        def _eval_fn(model, batch, batch_size, device):
            img = batch['img'].to(device).to(torch.float32) / 127.5 - 1
            preds = model(img)
            return preds
        return _eval_fn


    def _init_modules(self):
        cfg = self.cfg

        # build model
        print_(self.rank, '[INFO] initializing model...')
        self.model = build_model(
            cfg,
            cfg.model,
        ).train().requires_grad_(False).to(self.device)

        # resume training
        assert not cfg.resume


    def _init_phases(self):
        print_(self.rank, '[INFO] initializing training phases...')
        self.phases = [self._build_loss_phase('main', self.model)]


    def get_modules_for_save(self):
        return [('model', self.model)]


    def _get_modules_for_distribution(self):
        return [('model', self.model, True)]


    def print_models(self):
        imsize = self.cfg.dataset.imsize
        input = torch.empty(
            [self.batch_gpu, 3, imsize, imsize],
            device=self.device,
        )
        _ = print_module_summary(self.model, [input])


class LatentGenderClassificationTrainer(BaseTrainer):
    def run_batch(self, batch, batch_idx, cur_step):
        # prep_data
        with torch.autograd.profiler.record_function('data_prep'):
            phase_img = batch['img']
            phase_img = (
                phase_img.to(self.device).to(torch.float32) / 127.5 - 1
            ).split(self.batch_gpu)

            phase_gender = batch['gender']
            phase_gender = (
                phase_gender.to(self.device).to(torch.float32)
            ).split(self.batch_gpu)

        # run training phases
        for phase in self.phases:
            if batch_idx % phase.interval != 0:
                continue

            phase.init_gradient_accumulation()

            # accumulate gradients over multiple rounds
            for round_idx, (img, gender) in enumerate(zip(
                phase_img,
                phase_gender,
            )):
                sync = (round_idx == self.batch_size // \
                    (self.batch_gpu * self.num_gpus) - 1)
                gain = phase.interval
                self.loss.accumulate_gradients(
                    phase=phase.name,
                    img=img,
                    gender=gender,
                    sync=sync,
                    gain=gain,
                )

            phase.update_params()


    def run_C(self, img, sync):
        C_e = self.ddp_modules['C_e']
        C_c = self.ddp_modules['C_c']

        # encode
        with misc.ddp_sync(C_e, sync):
            enc = C_e(img)

        # classify
        with misc.ddp_sync(C_c, sync):
            pred = C_c(enc)

        return pred


    def get_eval_fn(self):
        def _eval_fn(model, batch, batch_size, device):
            img = batch['img'].to(device).to(torch.float32) / 127.5 - 1
            target = batch['gender'].to(device).to(torch.float32)
            pred = model(img)
            return pred, target
        return _eval_fn


    def _init_modules(self):
        cfg = self.cfg

        # build model
        print_(self.rank, '[INFO] initializing model...')
        self.C = build_model(
            cfg,
            cfg.model,
        ).train().requires_grad_(False).to(self.device)

        # resume training
        assert not cfg.resume


    def _init_phases(self):
        print_(self.rank, '[INFO] initializing training phases...')
        self.phases = [self._build_loss_phase('classify', self.C)]


    def get_modules_for_save(self):
        return [('C', self.C)]


    def _get_modules_for_distribution(self):
        return [
            ('C_e', self.C.e, False),
            ('C_c', self.C.c, True),
        ]


    def print_models(self):
        input = torch.empty(
            [self.batch_gpu, 3, self.C.imsize, self.C.imsize],
            device=self.device,
        )
        _ = print_module_summary(self.C, [input])
