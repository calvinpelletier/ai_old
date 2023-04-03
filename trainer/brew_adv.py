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


class BaseAdvBrewTrainer(StyleGanTrainer):
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

        self._update_model_ema(cur_step)

        self._run_ada_heuristic(batch_idx)


    def _init_modules(self):
        print_(self.rank, '[INFO] initializing models...')
        cfg = self.cfg

        from_scratch = not cfg.resume
        assert from_scratch, 'todo'

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

        # build/load discriminator
        if cfg.model.D.type == 'dep':
            self.D, _ = build_model_from_exp(cfg.model.D.exp, 'D')
            self.D = self.D.train().requires_grad_(False).to(self.device)
        else:
            self.D = build_model(
                cfg,
                cfg.model.D,
            ).train().requires_grad_(False).to(self.device)

        # face loss
        self.face_loss_model = FaceIdLoss(
            cfg.dataset.imsize).eval().requires_grad_(False).to(self.device)

        # perceptual loss
        perceptual_type = cfg.loss.G.rec.perceptual_type
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

        # model ema
        self.G_ema = copy.deepcopy(self.G).eval()
        self.ema_ksteps = 10
        self.ema_rampup = 0.05 if from_scratch else None

        # resume training
        assert not cfg.resume

        # augmentation
        print_(self.rank, '[INFO] initializing augmentation...')
        self.augment_pipe = None
        self.ada_stats = None
        if cfg.trainer.aug.enabled:
            self.augment_pipe = build_augpipe(
                cfg.trainer.aug,
            ).train().requires_grad_(False).to(self.device)
            self.augment_pipe.p.copy_(torch.as_tensor(0))
            if cfg.trainer.aug.target is not None:
                self.ada_stats = training_stats.Collector(
                    regex='Loss/signs/real')
            if cfg.trainer.aug.speed == 'auto':
                cfg.trainer.aug.speed = 500 if from_scratch else 100


    def _init_phases(self):
        print_(self.rank, '[INFO] initializing training phases...')
        cfg = self.cfg
        self.phases = []

        # generator phase
        self.phases.append(TrainingPhase(
            name='G_aae',
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


    def _get_modules_for_distribution(self):
        return [
            ('teacher', self.teacher, False),
            ('G_e', self.G.e, True),
            ('G_g', self.G.g, True),
            ('D', self.D, True),
            (None, self.G_ema, False),
            ('augment_pipe', self.augment_pipe, True),
            ('perceptual', self.perceptual_loss_model, False),
            ('arcface', self.face_loss_model, False),
        ]


    def get_model_key_for_eval(self):
        return 'G_ema'


    def print_models(self):
        imsize = self.cfg.dataset.imsize
        img = torch.empty(
            [self.batch_gpu, 3, imsize, imsize],
            device=self.device,
        )
        gender = torch.empty([self.batch_gpu], device=self.device)
        rec = print_module_summary(self.G, [img, gender, 1.])
        print_module_summary(self.D, [rec])


class AdvBrewTrainerV2(BaseAdvBrewTrainer):
    def get_target_img(self, w, gender, sync):
        teacher = self.ddp_modules['teacher']
        with misc.ddp_sync(teacher, sync):
            target = teacher(w, gender, random() * MAX_MAG)
        if self.cfg.trainer.xflip and random() > 0.5:
            return target.flip(3)
        return target


    def run_G(self, w, gender, do_rec_losses, sync):
        teacher = self.ddp_modules['teacher']
        G_g = self.ddp_modules['G_g']
        G_e = self.ddp_modules['G_e']
        xflip = self.cfg.trainer.xflip and random() > 0.5

        # teacher
        with misc.ddp_sync(teacher, sync):
            base = teacher(w, gender, 0.)
            if do_rec_losses:
                target1 = teacher(w, gender, 0.5)
                target2 = teacher(w, gender, 1.)
                target3 = teacher(w, gender, 1.5)
            if xflip:
                base = base.flip(3)
                if do_rec_losses:
                    target1 = target1.flip(3)
                    target2 = target2.flip(3)
                    target3 = target3.flip(3)

        # encode
        with misc.ddp_sync(G_e, sync):
            idt, z, delta = G_e(base.detach(), gender)

        # generate
        with misc.ddp_sync(G_g, sync):
            if do_rec_losses:
                output0 = G_g(idt, z, delta, 0.)
                output1 = G_g(idt, z, delta, 0.5)
                output2 = G_g(idt, z, delta, 1.)
                output3 = G_g(idt, z, delta, 1.5)
            else:
                output = G_g(idt, z, delta, random() * MAX_MAG)

        if not do_rec_losses:
            return output

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


class AdvBrewTrainer(BaseAdvBrewTrainer):
    # TODO: speed up
    def get_target_img(self, w, gender, sync):
        teacher = self.ddp_modules['teacher']
        with misc.ddp_sync(teacher, sync):
            target, _, _ = teacher(w, gender, random() * MAX_MAG, 0., 0.)
        if self.cfg.trainer.xflip and random() > 0.5:
            return target.flip(3)
        return target


    def run_G(self, w, gender, do_rec_losses, sync):
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

        if not do_rec_losses:
            return output1 if random() > 0.5 else output2

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


class OldAdvBrewTrainer(BaseAdvBrewTrainer):
    def get_target_img(self, w, gender, sync):
        teacher = self.ddp_modules['teacher']
        with misc.ddp_sync(teacher, sync):
            target = teacher(w, gender, random() * MAX_MAG)
        if random() > 0.5:
            return target.flip(3)
        return target


    def run_G(self, w, gender, do_rec_losses, sync):
        teacher = self.ddp_modules['teacher']
        G = self.ddp_modules['G']
        mag = random() * MAX_MAG
        xflip = random() > 0.5

        with misc.ddp_sync(teacher, sync):
            base = teacher(w, gender, 0.)
            if xflip:
                base = base.flip(3)
            if do_rec_losses:
                target = teacher(w, gender, mag)
                if xflip:
                    target = target.flip(3)

        with misc.ddp_sync(G, sync):
            output = G(base.detach(), gender, mag)

        if not do_rec_losses:
            return output

        # face loss
        face_loss_model = self.ddp_modules['arcface']
        with misc.ddp_sync(face_loss_model, sync):
            face_loss = face_loss_model(output, target.detach())

        # perceptual loss
        perceptual_loss_model = self.ddp_modules['perceptual']
        with misc.ddp_sync(perceptual_loss_model, sync):
            perceptual_loss = perceptual_loss_model(output, target.detach())

        return target, output, face_loss, perceptual_loss


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
