#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from ai_old.util.etc import print_
from ai_old.trainer.base import BaseTrainer
from external.sg2.misc import print_module_summary
from ai_old.util.factory import build_model
import external.sg2.misc as misc
from ai_old.loss.perceptual.face import FaceIdLoss
from ai_old.loss.clip import ClipLoss, MultiTextClipLoss
from ai_old.loss.ss import SynthSwapDeltaLoss
from ai_old.loss.gender import GenderLoss


class WPlusLerpTrainer(BaseTrainer):
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
        G_g = self.ddp_modules['G_g']
        G_f = self.ddp_modules['G_f']

        # lerp
        with misc.ddp_sync(G_f, sync):
            new_w = G_f(w, gender, magnitude=1.)

        # generate
        with misc.ddp_sync(G_g, sync):
            new_img = G_g(new_w)

        ret = {
            'new_w': new_w,
            'new_img': new_img,
        }

        # face id loss
        if self.face_loss_enabled:
            face_loss_model = self.ddp_modules['arcface']
            with misc.ddp_sync(face_loss_model, sync):
                face_loss = face_loss_model(new_img, img)
            ret['face_loss'] = face_loss

        # clip loss
        if self.clip_loss_enabled or self.hair_clip_loss_enabled:
            clip_loss_model = self.ddp_modules['clip']
            with misc.ddp_sync(clip_loss_model, sync):
                clip_losses = clip_loss_model('all', new_img, gender)
                if 'main' in clip_losses:
                    ret['clip_loss'] = clip_losses['main']
                if 'hair' in clip_losses:
                    ret['hair_clip_loss'] = clip_losses['hair']

        # ss delta loss
        if self.ss_delta_loss_enabled:
            ss_delta_loss_model = self.ddp_modules['ss_delta']
            with misc.ddp_sync(ss_delta_loss_model, sync):
                ss_delta_loss = ss_delta_loss_model(new_w, w, gender)
            ret['ss_delta_loss'] = ss_delta_loss

        # classification loss
        if self.classify_loss_enabled:
            classify_loss_model = self.ddp_modules['classify']
            with misc.ddp_sync(classify_loss_model, sync):
                classify_loss = classify_loss_model(new_img, 1. - gender)
            ret['classify_loss'] = classify_loss

        return ret


    def get_eval_fn(self):
        def _eval_fn(G, batch, batch_size, device):
            img = batch['img'].to(device).to(torch.float32) / 127.5 - 1
            gender = batch['gender'].to(device).to(
                torch.float32).unsqueeze(1)
            w = batch['w'].to(device).to(torch.float32)
            new_w = G.f(w, gender, magnitude=1.)
            new_img = G.g(new_w)
            return img, gender, w, new_img, new_w
        return _eval_fn


    def get_sample_fn(self):
        def _sample_fn(model, batch, batch_size, device):
            gender = batch['gender'].to(device).to(
                torch.float32).unsqueeze(1)
            w = batch['w'].to(device).to(torch.float32)
            new_img = model(w, gender, magnitude=1.)
            new_img = (new_img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return new_img
        return _sample_fn


    def _init_modules(self):
        cfg = self.cfg

        self.gendered = hasattr(cfg, 'gendered') and cfg.gendered

        # build main model
        print_(self.rank, '[INFO] initializing model...')
        self.G = build_model(
            cfg,
            cfg.model.G,
        ).train().requires_grad_(False).to(self.device)

        # face loss
        self.face_loss_enabled = hasattr(cfg.loss, 'face') and \
            cfg.loss.face.weight > 0.
        if self.face_loss_enabled:
            self.face_loss_model = FaceIdLoss(
                cfg.dataset.imsize).eval().requires_grad_(False).to(self.device)
        else:
            self.face_loss_model = None

        # clip loss
        self.clip_loss_enabled = hasattr(cfg.loss, 'clip') and \
            cfg.loss.clip.weight > 0.
        self.hair_clip_loss_enabled = hasattr(cfg.loss, 'hair_clip') and \
            cfg.loss.hair_clip.weight > 0.
        if self.clip_loss_enabled or self.hair_clip_loss_enabled:
            target_texts = {}
            if self.clip_loss_enabled:
                target_texts['main'] = cfg.loss.clip.target_text
            if self.hair_clip_loss_enabled:
                target_texts['hair'] = cfg.loss.hair_clip.target_text

            assert self.gendered, 'TODO'
            self.clip_loss_model = MultiTextClipLoss(
                target_texts,
                self.device,
            ).eval().requires_grad_(False).to(self.device)
        else:
            self.clip_loss_model = None

        # ss delta loss
        self.ss_delta_loss_enabled = hasattr(cfg.loss, 'ss_delta') and \
            cfg.loss.ss_delta.weight > 0.
        if self.ss_delta_loss_enabled:
            self.ss_delta_loss_model = \
                SynthSwapDeltaLoss().eval().requires_grad_(False).to(
                    self.device)
        else:
            self.ss_delta_loss_model = None

        # classification loss
        self.classify_loss_enabled = hasattr(cfg.loss, 'classify') and \
            cfg.loss.classify.weight > 0.
        if self.classify_loss_enabled:
            self.classify_loss_model = GenderLoss(
                l2=cfg.loss.classify.l2,
            ).eval().requires_grad_(False).to(self.device)
        else:
            self.classify_loss_model = None

        # resume training
        assert not cfg.resume


    def _init_phases(self):
        print_(self.rank, '[INFO] initializing training phases...')
        self.phases = [self._build_loss_phase('main', self.G)]


    def get_modules_for_save(self):
        return [('G', self.G)]


    def _get_modules_for_distribution(self):
        ret = [
            ('G_g', self.G.g, False),
            ('G_f', self.G.f, True),
        ]
        if self.face_loss_enabled:
            ret.append(('arcface', self.face_loss_model, False))
        if self.clip_loss_enabled or self.hair_clip_loss_enabled:
            ret.append(('clip', self.clip_loss_model, False))
        if self.ss_delta_loss_enabled:
            ret.append(('ss_delta', self.ss_delta_loss_model, False))
        if self.classify_loss_enabled:
            ret.append(('classify', self.classify_loss_model, False))
        return ret


    def print_models(self):
        w_plus = torch.empty([self.batch_gpu, 18, 512], device=self.device)
        gender = torch.empty([self.batch_gpu, 1], device=self.device)
        _ = print_module_summary(self.G.f, [w_plus, gender])



class WPlusDualLerpTrainer(WPlusLerpTrainer):
    def run_G(self, img, gender, w, sync):
        G_g = self.ddp_modules['G_g']
        G_f1 = self.ddp_modules['G_f1']
        G_f2 = self.ddp_modules['G_f2']

        # lerp
        with misc.ddp_sync(G_f1, sync):
            w1 = G_f1(w, gender, magnitude=1.)
        with misc.ddp_sync(G_f2, sync):
            w2 = G_f2(w1, gender, magnitude=1.)

        # generate
        with misc.ddp_sync(G_g, sync):
            img1, seg1 = G_g(w1)
            img1 = img1.detach()
            seg1 = seg1.detach()
            img2, seg2 = G_g(w2)

        ret = {
            'w1': w1,
            'w2': w2,
            'img1': img1,
            'img2': img2,
            'seg1': seg1,
            'seg2': seg2,
        }

        # face id loss
        if self.face_loss_enabled:
            face_loss_model = self.ddp_modules['arcface']
            with misc.ddp_sync(face_loss_model, sync):
                ret['face_loss'] = face_loss_model(img2, img1)

        # clip loss
        if self.clip_loss_enabled or self.hair_clip_loss_enabled:
            clip_loss_model = self.ddp_modules['clip']
            with misc.ddp_sync(clip_loss_model, sync):
                clip_losses = clip_loss_model('all', img2, gender)
                if 'main' in clip_losses:
                    ret['clip_loss'] = clip_losses['main']
                if 'hair' in clip_losses:
                    ret['hair_clip_loss'] = clip_losses['hair']

        # ss delta loss
        if self.ss_delta_loss_enabled:
            ss_delta_loss_model = self.ddp_modules['ss_delta']
            with misc.ddp_sync(ss_delta_loss_model, sync):
                ret['ss_delta_loss'] = ss_delta_loss_model(w2, w1, gender)

        return ret


    def get_eval_fn(self):
        def _eval_fn(G, batch, batch_size, device):
            img = batch['img'].to(device).to(torch.float32) / 127.5 - 1
            gender = batch['gender'].to(device).to(
                torch.float32).unsqueeze(1)
            w = batch['w'].to(device).to(torch.float32)
            w1 = G.f1(w, gender, magnitude=1.)
            w2 = G.f2(w1, gender, magnitude=1.)
            img1, seg1 = G.g(w1)
            img2, seg2 = G.g(w2)
            return img, gender, w, w1, w2, img1, img2, seg1, seg2
        return _eval_fn


    def get_sample_fn(self):
        def _sample_fn(model, batch, batch_size, device):
            gender = batch['gender'].to(device).to(
                torch.float32).unsqueeze(1)
            w = batch['w'].to(device).to(torch.float32)
            img1, img2 = model(w, gender, magnitude=1.)
            return img1, img2
        return _sample_fn


    def _get_modules_for_distribution(self):
        ret = [
            ('G_g', self.G.g, False),
            ('G_f1', self.G.f1, False),
            ('G_f2', self.G.f2, True),
        ]
        if self.face_loss_enabled:
            ret.append(('arcface', self.face_loss_model, False))
        if self.clip_loss_enabled or self.hair_clip_loss_enabled:
            ret.append(('clip', self.clip_loss_model, False))
        if self.ss_delta_loss_enabled:
            ret.append(('ss_delta', self.ss_delta_loss_model, False))
        if self.classify_loss_enabled:
            ret.append(('classify', self.classify_loss_model, False))
        return ret


    def print_models(self):
        w_plus = torch.empty([self.batch_gpu, 18, 512], device=self.device)
        gender = torch.empty([self.batch_gpu, 1], device=self.device)
        _ = print_module_summary(self.G.f2, [w_plus, gender])
