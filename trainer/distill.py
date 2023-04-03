#!/usr/bin/env python3
import torch
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


class Sg2DistillTrainer(StyleGanTrainer):
    def run_batch(self, batch, batch_idx, cur_step):
        # prep_data
        with torch.autograd.profiler.record_function('data_prep'):
            if random() > 0.5:
                img_key = 'img1'
                w_key = 'w1'
            else:
                img_key = 'img2'
                w_key = 'w2'

            phase_target = batch[img_key]
            phase_target = (
                phase_target.to(self.device).to(torch.float32) / 127.5 - 1
            ).split(self.batch_gpu)

            phase_w = batch[w_key]
            phase_w = (
                phase_w.to(self.device).to(torch.float32)
            ).split(self.batch_gpu)

        # run training phases
        for phase in self.phases:
            if batch_idx % phase.interval != 0:
                continue

            phase.init_gradient_accumulation()

            # accumulate gradients over multiple rounds
            for round_idx, (target, w) in enumerate(zip(
                phase_target,
                phase_w,
            )):
                sync = (round_idx == self.batch_size // \
                    (self.batch_gpu * self.num_gpus) - 1)
                gain = phase.interval
                self.loss.accumulate_gradients(
                    phase=phase.name,
                    target=target,
                    w=w,
                    sync=sync,
                    gain=gain,
                )

            phase.update_params()

        self._update_model_ema(cur_step)

        self._run_ada_heuristic(batch_idx)


    def run_G(self, target, w, do_rec_losses, sync):
        if self.freeze_first_half:
            G_g1 = self.ddp_modules['G_g1']
            with misc.ddp_sync(G_g1, sync):
                intermediate = G_g1(w)

            G_g2 = self.ddp_modules['G_g2']
            with misc.ddp_sync(G_g2, sync):
                output = G_g2(intermediate)
        else:
            G = self.ddp_modules['G']
            with misc.ddp_sync(G, sync):
                output = G(w)

        if not do_rec_losses:
            return output

        # face loss
        face_loss_model = self.ddp_modules['arcface']
        with misc.ddp_sync(face_loss_model, sync):
            face_loss = face_loss_model(output, target)

        # perceptual loss
        perceptual_loss_model = self.ddp_modules['perceptual']
        with misc.ddp_sync(perceptual_loss_model, sync):
            perceptual_loss = perceptual_loss_model(output, target)

        return output, face_loss, perceptual_loss


    def get_eval_fn(self):
        def eval_fn(model, batch, batch_size, device):
            target1 = batch['img1'].to(device).to(torch.float32) / 127.5 - 1
            w1 = batch['w1'].to(device).to(torch.float32)
            rec = model(w1)
            # rec = (rec * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return target1, rec
        return eval_fn


    def get_sample_fn(self):
        def sample_fn(model, batch, batch_size, device):
            target1 = batch['img1'].to(device).to(torch.float32) / 127.5 - 1
            w1 = batch['w1'].to(device).to(torch.float32)
            target2 = batch['img2'].to(device).to(torch.float32) / 127.5 - 1
            w2 = batch['w2'].to(device).to(torch.float32)
            rec1 = model(w1)
            rec2 = model(w2)
            # rec1 = (rec1 * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            # rec2 = (rec2 * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return rec1, rec2
        return sample_fn


    def _init_modules(self):
        print_(self.rank, '[INFO] initializing models...')
        cfg = self.cfg

        from_scratch = not cfg.resume
        assert from_scratch, 'todo'

        self.freeze_first_half = cfg.trainer.freeze_first_half

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
        ret = [
            ('D', self.D, True),
            (None, self.G_ema, False),
            ('augment_pipe', self.augment_pipe, True),
            ('perceptual', self.perceptual_loss_model, False),
            ('arcface', self.face_loss_model, False),
        ]
        if self.freeze_first_half:
            ret.append(('G_g1', self.G.g1, False))
            ret.append(('G_g2', self.G.g2, True))
        else:
            ret.append(('G', self.G, True))
        return ret


    def get_model_key_for_eval(self):
        return 'G_ema'


    def print_models(self):
        w = torch.empty(
            [self.batch_gpu, 18, 512],
            device=self.device,
        )
        rec = print_module_summary(self.G, [w])
        print_module_summary(self.D, [rec])
