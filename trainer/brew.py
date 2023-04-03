#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from ai_old.util.etc import print_
from ai_old.trainer.gan import StyleGanTrainer
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
from random import random
from ai_old.nn.models.facegen.brew import Teacher
from ai_old.trainer.base import BaseTrainer

# tmp
from ai_old.util.etc import normalized_tensor_to_pil_img


MAX_MAG = 1.5


class BaseBrewTrainer(BaseTrainer):
    def run_batch(self, batch, batch_idx, cur_step):
        # prep_data
        with torch.autograd.profiler.record_function('data_prep'):
            phase_w = (
                batch['w'].to(self.device).to(torch.float32)
            ).split(self.batch_gpu)

            phase_gender = (
                batch['gender'].to(self.device).to(torch.float32)
            ).split(self.batch_gpu)

        # run training phases
        for phase in self.phases:
            if batch_idx % phase.interval != 0:
                continue

            phase.init_gradient_accumulation()

            # accumulate gradients over multiple rounds
            for round_idx, (w, gender) in enumerate(zip(phase_w, phase_gender)):
                sync = (round_idx == self.batch_size // \
                    (self.batch_gpu * self.num_gpus) - 1)
                gain = phase.interval
                self.loss.accumulate_gradients(
                    phase=phase.name,
                    w=w,
                    gender=gender,
                    sync=sync,
                    gain=gain,
                )

            phase.update_params()


    def run_G(self, w, gender, sync):
        pass


    def get_eval_fn(self):
        def eval_fn(model, batch, batch_size, device):
            w = batch['w'].to(device).to(torch.float32)
            gender = batch['gender'].to(device).to(torch.float32)
            mag = random() * MAX_MAG
            base = self.teacher(w, gender, 0.)
            target = self.teacher(w, gender, mag)
            rec = model(base, gender, mag)
            return target, rec
        return eval_fn


    def get_sample_fn(self):
        def sample_fn(model, batch, batch_size, device):
            w = batch['w'].to(device).to(torch.float32)
            gender = batch['gender'].to(device).to(torch.float32)
            base = self.teacher(w, gender, 0.)
            rec1 = model(base, gender, 0.)
            rec2 = model(base, gender, 1.)
            return rec1, rec2
        return sample_fn


    def _init_modules(self):
        print_(self.rank, '[INFO] initializing models...')
        cfg = self.cfg
        assert not cfg.resume, 'todo'

        # build teacher
        self.teacher = build_model(
            cfg,
            cfg.model.teacher,
        ).eval().requires_grad_(False).to(self.device)

        # build student
        self.G = build_model(
            cfg,
            cfg.model.G,
        ).train().requires_grad_(False).to(self.device)

        # face loss
        if cfg.loss.G.rec.face_enabled:
            self.face_loss_model = FaceIdLoss(
                cfg.dataset.imsize).eval().requires_grad_(False).to(self.device)
        else:
            self.face_loss_model = None

        # perceptual loss
        perceptual_type = cfg.loss.G.rec.perceptual_type
        if perceptual_type is not None:
            if perceptual_type == 'lpips':
                raise Exception('lpips_alex or lpips_vgg')
            elif perceptual_type == 'lpips_alex':
                percep = lpips.LPIPS(net='alex')
            elif perceptual_type == 'lpips_vgg':
                percep = lpips.LPIPS(net='vgg')
            elif perceptual_type == 'trad':
                percep = PerceptualLoss()
            elif perceptual_type == 'ahanu':
                percep = AhanuPercepLoss(cfg.loss.G.rec.perceptual_version)
            else:
                raise Exception(perceptual_type)
            percep = percep.eval().requires_grad_(False).to(self.device)
            self.perceptual_loss_model = percep
        else:
            self.perceptual_loss_model = None


    def _init_phases(self):
        print_(self.rank, '[INFO] initializing training phases...')
        self.phases = [self._build_loss_phase('main', self.G)]


    def get_modules_for_save(self):
        return [('G', self.G)]


    def _get_modules_for_distribution(self):
        ret = [
            ('teacher', self.teacher, False),
            ('G_e', self.G.e, True),
            ('G_g', self.G.g, True),
        ]
        if self.perceptual_loss_model is not None:
            ret.append(('perceptual', self.perceptual_loss_model, False))
        if self.face_loss_model is not None:
            ret.append(('arcface', self.face_loss_model, False))
        return ret


    def get_model_key_for_eval(self):
        return 'G'


    def print_models(self):
        imsize = self.cfg.dataset.imsize
        img = torch.empty(
            [self.batch_gpu, 3, imsize, imsize],
            device=self.device,
        )
        gender = torch.empty([self.batch_gpu], device=self.device)
        _ = print_module_summary(self.G, [img, gender, 1.])



class BinaryBrewTrainer(BaseBrewTrainer):
    def run_G(self, w, gender, sync):
        teacher = self.ddp_modules['teacher']
        G_g = self.ddp_modules['G_g']
        G_e = self.ddp_modules['G_e']
        xflip = self.cfg.trainer.xflip and random() > 0.5

        # teacher
        with misc.ddp_sync(teacher, sync):
            base = teacher(w, gender, 0.)
            target = teacher(w, gender, 1.)
            if xflip:
                base = base.flip(3)
                target = target.flip(3)

        # encode
        with misc.ddp_sync(G_e, sync):
            idt, z, delta = G_e(base.detach(), gender)

        # generate
        with misc.ddp_sync(G_g, sync):
            output0 = G_g(idt, z, delta, 0.)
            output1 = G_g(idt, z, delta, 1.)

        # face loss
        if 'arcface' in self.ddp_modules:
            face_loss_model = self.ddp_modules['arcface']
            with misc.ddp_sync(face_loss_model, sync):
                face_loss0 = face_loss_model(output0, base.detach())
                face_loss1 = face_loss_model(output1, target.detach())
                face_loss = face_loss0.mean() + face_loss1.mean()
        else:
            face_loss = None

        # perceptual loss
        if 'perceptual' in self.ddp_modules:
            percep_loss_model = self.ddp_modules['perceptual']
            with misc.ddp_sync(percep_loss_model, sync):
                percep_loss0 = percep_loss_model(output0, base.detach())
                percep_loss1 = percep_loss_model(output1, target.detach())
                perceptual_loss = percep_loss0.mean() + percep_loss1.mean()
        else:
            perceptual_loss = None

        # pixel loss
        pixel_loss0 = F.mse_loss(output0, base.detach())
        pixel_loss1 = F.mse_loss(output1, target.detach())
        pixel_loss = pixel_loss0.mean() + pixel_loss1.mean()

        return pixel_loss, face_loss, perceptual_loss


class BrewTrainerV2(BaseBrewTrainer):
    def run_G(self, w, gender, sync):
        teacher = self.ddp_modules['teacher']
        G_g = self.ddp_modules['G_g']
        G_e = self.ddp_modules['G_e']
        xflip = self.cfg.trainer.xflip and random() > 0.5

        # teacher
        with misc.ddp_sync(teacher, sync):
            base = teacher(w, gender, 0.)
            target1 = teacher(w, gender, 0.5)
            target2 = teacher(w, gender, 1.)
            target3 = teacher(w, gender, 1.5)
            if xflip:
                base = base.flip(3)
                target1 = target1.flip(3)
                target2 = target2.flip(3)
                target3 = target3.flip(3)

        # encode
        with misc.ddp_sync(G_e, sync):
            idt, z, delta = G_e(base.detach(), gender)

        # generate
        with misc.ddp_sync(G_g, sync):
            output0 = G_g(idt, z, delta, 0.)
            output1 = G_g(idt, z, delta, 0.5)
            output2 = G_g(idt, z, delta, 1.)
            output3 = G_g(idt, z, delta, 1.5)

        # face loss
        if 'arcface' in self.ddp_modules:
            face_loss_model = self.ddp_modules['arcface']
            with misc.ddp_sync(face_loss_model, sync):
                face_loss0 = face_loss_model(output0, base.detach())
                face_loss1 = face_loss_model(output1, target1.detach())
                face_loss2 = face_loss_model(output2, target2.detach())
                face_loss3 = face_loss_model(output3, target3.detach())
                face_loss = face_loss0.mean() + face_loss1.mean() + \
                    face_loss2.mean() + face_loss3.mean()
        else:
            face_loss = None

        # perceptual loss
        if 'perceptual' in self.ddp_modules:
            percep_loss_model = self.ddp_modules['perceptual']
            with misc.ddp_sync(percep_loss_model, sync):
                percep_loss0 = percep_loss_model(output0, base.detach())
                percep_loss1 = percep_loss_model(output1, target1.detach())
                percep_loss2 = percep_loss_model(output2, target2.detach())
                percep_loss3 = percep_loss_model(output3, target3.detach())
                perceptual_loss = percep_loss0.mean() + percep_loss1.mean() + \
                    percep_loss2.mean() + percep_loss3.mean()
        else:
            perceptual_loss = None

        # pixel loss
        pixel_loss0 = F.mse_loss(output0, base.detach())
        pixel_loss1 = F.mse_loss(output1, target1.detach())
        pixel_loss2 = F.mse_loss(output2, target2.detach())
        pixel_loss3 = F.mse_loss(output3, target3.detach())
        pixel_loss = pixel_loss0.mean() + pixel_loss1.mean() + \
            pixel_loss2.mean() + pixel_loss3.mean()

        return pixel_loss, face_loss, perceptual_loss


class BrewTrainer(BaseBrewTrainer):
    def run_G(self, w, gender, sync):
        teacher = self.ddp_modules['teacher']
        G_g = self.ddp_modules['G_g']
        G_e = self.ddp_modules['G_e']
        mag1 = random() * MAX_MAG
        mag2 = random() * MAX_MAG
        xflip = self.cfg.trainer.xflip and random() > 0.5

        # teacher
        with misc.ddp_sync(teacher, sync):
            base, target1, target2 = teacher(w, gender, 0., mag1, mag2)
            if xflip:
                base = base.flip(3)
                target1 = target1.flip(3)
                target2 = target2.flip(3)

        # tmp
        # normalized_tensor_to_pil_img(base[0]).save(
        #     '/home/asiu/data/tmp/debug/base.png')
        # normalized_tensor_to_pil_img(target1[0]).save(
        #     '/home/asiu/data/tmp/debug/target1.png')
        # normalized_tensor_to_pil_img(target2[0]).save(
        #     '/home/asiu/data/tmp/debug/target2.png')
        # raise Exception('a')

        # encode
        with misc.ddp_sync(G_e, sync):
            idt, z, delta = G_e(base.detach(), gender)

        # generate
        with misc.ddp_sync(G_g, sync):
            output1 = G_g(idt, z, delta, mag1)
            output2 = G_g(idt, z, delta, mag2)

        # face loss
        if 'arcface' in self.ddp_modules:
            face_loss_model = self.ddp_modules['arcface']
            with misc.ddp_sync(face_loss_model, sync):
                face_loss1 = face_loss_model(output1, target1.detach())
                face_loss2 = face_loss_model(output2, target2.detach())
                face_loss = face_loss1.mean() + face_loss2.mean()
        else:
            face_loss = None

        # perceptual loss
        if 'perceptual' in self.ddp_modules:
            percep_loss_model = self.ddp_modules['perceptual']
            with misc.ddp_sync(percep_loss_model, sync):
                percep_loss1 = percep_loss_model(output1, target1.detach())
                percep_loss2 = percep_loss_model(output2, target2.detach())
                perceptual_loss = percep_loss1.mean() + percep_loss2.mean()
        else:
            perceptual_loss = None

        # pixel loss
        pixel_loss1 = F.mse_loss(output1, target1.detach())
        pixel_loss2 = F.mse_loss(output2, target2.detach())
        pixel_loss = pixel_loss1 + pixel_loss2

        return pixel_loss, face_loss, perceptual_loss


    def get_eval_fn(self):
        def eval_fn(model, batch, batch_size, device):
            w = batch['w'].to(device).to(torch.float32)
            gender = batch['gender'].to(device).to(torch.float32)
            mag = random() * MAX_MAG
            base, target, _ = self.teacher(w, gender, 0., mag, mag)
            rec = model(base, gender, mag)
            return target, rec
        return eval_fn


    def get_sample_fn(self):
        def sample_fn(model, batch, batch_size, device):
            w = batch['w'].to(device).to(torch.float32)
            gender = batch['gender'].to(device).to(torch.float32)
            base, _, _ = self.teacher(w, gender, 0., 0., 0.)
            rec1 = model(base, gender, 0.)
            rec2 = model(base, gender, 1.)
            return rec1, rec2
        return sample_fn


class DebugBrewTrainer(BaseBrewTrainer):
    def run_G(self, w, gender, sync):
        teacher = self.ddp_modules['teacher']
        G_g = self.ddp_modules['G_g']
        G_e = self.ddp_modules['G_e']

        # teacher
        with misc.ddp_sync(teacher, sync):
            base = teacher(w, gender, 0.)

        # encode
        with misc.ddp_sync(G_e, sync):
            idt, z, delta = G_e(base.detach(), gender)

        # generate
        with misc.ddp_sync(G_g, sync):
            output = G_g(idt, z, delta, 0.)

        # pixel loss
        pixel_loss = F.mse_loss(output, base.detach())

        return pixel_loss, None, None
