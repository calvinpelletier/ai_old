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
from ai_old.loss.perceptual.face import FaceLoss
from ai_old.loss.perceptual.lpips import LpipsLoss
from ai_old.loss.perceptual.trad import PerceptualLoss
from external.optimizer import get_optimizer
from ai_old.trainer.phase import TrainingPhase


class E4eTrainer(StyleGanTrainer):
    def run_batch(self, batch, batch_idx, cur_step):
        # prep_data
        with torch.autograd.profiler.record_function('data_prep'):
            phase_real = batch['y']
            phase_real = (
                phase_real.to(self.device).to(torch.float32) / 127.5 - 1
            ).split(self.batch_gpu)

        # run training phases
        for phase in self.phases:
            if batch_idx % phase.interval != 0:
                continue

            phase.init_gradient_accumulation()

            # accumulate gradients over multiple rounds
            for round_idx, real in enumerate(phase_real):
                sync = (round_idx == self.batch_size // \
                    (self.batch_gpu * self.num_gpus) - 1)
                gain = phase.interval
                self.loss.accumulate_gradients(
                    phase=phase.name,
                    real=real,
                    sync=sync,
                    gain=gain,
                )

            phase.update_params()

        self._update_model_ema(cur_step)

        self._run_ada_heuristic(batch_idx)


    def run_G(self, real, do_req_losses, sync):
        G_g = self.ddp_modules['G_g']
        G_e = self.ddp_modules['G_e']
        face_loss_model = self.ddp_modules['arcface']
        perceptual_loss_model = self.ddp_modules['perceptual']

        if self.G.intermediate == 'enc_plus_z':
            # encode
            with misc.ddp_sync(G_e, sync):
                encoding, z = G_e(real)

            # synthesize
            with misc.ddp_sync(G_g, sync):
                output = G_g(encoding, z)
        elif self.G.intermediate == 'modspace' or self.G.intermediate == 'enc':
            # encode
            with misc.ddp_sync(G_e, sync):
                ws = G_e(real)

            # synthesize
            with misc.ddp_sync(G_g, sync):
                output = G_g(ws)
        else:
            raise Exception(self.G.intermediate)

        if not do_req_losses:
            return output

        # face loss
        with misc.ddp_sync(face_loss_model, sync):
            face_loss = face_loss_model(output, real)

        # perceptual loss
        with misc.ddp_sync(perceptual_loss_model, sync):
            perceptual_loss = perceptual_loss_model(output, real)

        return output, face_loss, perceptual_loss


    def get_eval_fn(self):
        def eval_fn(model, batch, batch_size, device):
            real = batch['y'].to(device).to(torch.float32) / 127.5 - 1
            rec = model(real)
            # rec = (rec * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return real, rec
        return eval_fn


    def get_sample_fn(self):
        def sample_fn(model, batch, batch_size, device):
            real = batch['y'].to(device).to(torch.float32) / 127.5 - 1
            rec = model(real)
            rec = (rec * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return rec
        return sample_fn


    def _init_modules(self):
        print_(self.rank, '[INFO] initializing models...')
        cfg = self.cfg

        # TODO: initialized aae should be considered not from scratch
        from_scratch = not cfg.resume

        # build/load generator
        if cfg.model.G.type == 'dep':
            self.G, _ = build_model_from_exp(cfg.model.G.exp, 'G')
            self.G = self.G.train().requires_grad_(False).to(self.device)
        else:
            self.G = build_model(
                cfg,
                cfg.model.G,
            ).train().requires_grad_(False).to(self.device)

        # load discriminator
        if cfg.model.D.type == 'dep':
            self.D, _ = build_model_from_exp(cfg.model.D.exp, 'D')
            self.D = self.D.train().requires_grad_(False).to(self.device)
        else:
            self.D = build_model(
                cfg,
                cfg.model.D,
            ).train().requires_grad_(False).to(self.device)

        # face loss
        self.face_loss_model = FaceLoss().eval().requires_grad_(False).to(
            self.device)

        # perceptual loss
        perceptual_type = cfg.loss.G.rec.perceptual_type
        assert perceptual_type in ['lpips', 'trad']
        percep = LpipsLoss() if perceptual_type == 'lpips' else \
            PerceptualLoss()
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
            ('G_g', self.G.g, True),
            ('G_e', self.G.e, True),
            ('D', self.D, True),
            (None, self.G_ema, False),
            ('augment_pipe', self.augment_pipe, True),
            ('arcface', self.face_loss_model, False),
            ('perceptual', self.perceptual_loss_model, False),
        ]


    def print_models(self):
        real = torch.empty(
            [self.batch_gpu, 3, self.G.imsize, self.G.imsize],
            device=self.device,
        )
        rec = print_module_summary(self.G, [real])
        print_module_summary(self.D, [rec])



class BlendAaeTrainer(AaeTrainer):
    def run_batch(self, batch, batch_idx, cur_step):
        # prep_data
        with torch.autograd.profiler.record_function('data_prep'):
            phase_full = batch['full']
            phase_full = (
                phase_full.to(self.device).to(torch.float32) / 127.5 - 1
            ).split(self.batch_gpu)

            phase_fg = batch['fg']
            phase_fg = (
                phase_fg.to(self.device).to(torch.float32) / 127.5 - 1
            ).split(self.batch_gpu)

            phase_ibg = batch['ibg']
            phase_ibg = (
                phase_ibg.to(self.device).to(torch.float32) / 127.5 - 1
            ).split(self.batch_gpu)

        # run training phases
        for phase in self.phases:
            if batch_idx % phase.interval != 0:
                continue

            phase.init_gradient_accumulation()

            # accumulate gradients over multiple rounds
            for round_idx, (full, fg, ibg) in enumerate(zip(
                phase_full,
                phase_fg,
                phase_ibg,
            )):
                sync = (round_idx == self.batch_size // \
                    (self.batch_gpu * self.num_gpus) - 1)
                gain = phase.interval
                self.loss.accumulate_gradients(
                    phase=phase.name,
                    full=full,
                    fg=fg,
                    ibg=ibg,
                    sync=sync,
                    gain=gain,
                )

            phase.update_params()

        self.__update_model_ema(cur_step)

        self.__run_ada_heuristic(batch_idx)


    def run_G(self, full, fg, ibg, sync):
        G_g = self.ddp_modules['G_g']
        G_e = self.ddp_modules['G_e']
        face_loss_model = self.ddp_modules['arcface']
        lpips_loss_model = self.ddp_modules['lpips']

        # encode
        with misc.ddp_sync(G_e, sync):
            ws = G_e(fg)

        # synthesize
        with misc.ddp_sync(G_g, sync):
            output = G_g(ws, ibg)

        # face loss
        with misc.ddp_sync(face_loss_model, sync):
            face_loss = face_loss_model(output, full)

        # lpips loss
        with misc.ddp_sync(lpips_loss_model, sync):
            lpips_loss = lpips_loss_model(output, full)

        return output, face_loss, lpips_loss


    def get_eval_fn(self):
        def eval_fn(model, batch, batch_size, device):
            fg = batch['fg'].to(device).to(torch.float32) / 127.5 - 1
            ibg = batch['ibg'].to(device).to(torch.float32) / 127.5 - 1
            rec = model(fg, ibg)
            rec = (rec * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return rec
        return eval_fn


    def get_sample_fn(self):
        def sample_fn(model, batch, batch_size, device):
            fg = batch['fg'].to(device).to(torch.float32) / 127.5 - 1
            ibg = batch['ibg'].to(device).to(torch.float32) / 127.5 - 1
            rec = model(fg, ibg)
            rec = (rec * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return rec
        return sample_fn


    def print_models(self):
        fg = torch.empty(
            [self.batch_gpu, 3, self.G.imsize, self.G.imsize],
            device=self.device,
        )
        ibg = torch.empty(
            [self.batch_gpu, 3, self.G.imsize, self.G.imsize],
            device=self.device,
        )
        rec = print_module_summary(self.G, [fg, ibg])
        print_module_summary(self.D, [rec])
