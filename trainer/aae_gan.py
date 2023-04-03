#!/usr/bin/env python3
import torch
from ai_old.util.etc import print_
from ai_old.trainer.gan import StyleGanTrainer
from external.sg2.misc import print_module_summary
from ai_old.util.factory import build_model, build_augpipe
import copy
from external.sg2 import training_stats
import external.sg2.misc as misc


# TODO: fix training phase issue 
class BlendAaeGanComboTrainer(StyleGanTrainer):
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

            all_z = torch.randn(
                [len(self.phases) * self.batch_size, self.G.z_dims],
                device=self.device,
            )
            all_z = [
                phase_z.split(self.batch_gpu) \
                for phase_z in all_z.split(self.batch_size)
            ]

        # run training phases
        for phase, phase_z in zip(self.phases, all_z):
            if batch_idx % phase.interval != 0:
                continue

            phase.init_gradient_accumulation()

            # accumulate gradients over multiple rounds
            for round_idx, (full, fg, ibg, z) in enumerate(zip(
                phase_full,
                phase_fg,
                phase_ibg,
                phase_z,
            )):
                sync = (round_idx == self.batch_size // \
                    (self.batch_gpu * self.num_gpus) - 1)
                gain = phase.interval
                self.loss.accumulate_gradients(
                    phase=phase.name,
                    full=full,
                    fg=fg,
                    ibg=ibg,
                    z=z,
                    sync=sync,
                    gain=gain,
                )

            phase.update_params()

        self.__update_model_ema(cur_step)

        self.__run_ada_heuristic(batch_idx)


    def run_G_gen(self, ibg, z, sync):
        G_f = self.ddp_modules['G_f']
        G_g = self.ddp_modules['G_g']
        style_mix_prob = self.cfg.trainer.style_mix_prob

        # disentangle latent vector
        with misc.ddp_sync(G_f, sync):
            ws = G_f(z, None)
            if style_mix_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty(
                        [],
                        dtype=torch.int64,
                        device=ws.device,
                    ).random_(1, ws.shape[1])
                    cutoff = torch.where(
                        torch.rand([], device=ws.device) < style_mix_prob,
                        cutoff,
                        torch.full_like(cutoff, ws.shape[1]),
                    )
                    ws[:, cutoff:] = G_f(
                        torch.randn_like(z),
                        None,
                        skip_w_avg_update=True,
                    )[:, cutoff:]

        # synthesize image
        with misc.ddp_sync(G_g, sync):
            img = G_g(ws, ibg)

        return img, ws


    def run_G_rec(self, full, fg, ibg, sync):
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
        def eval_fn(model, _batch, batch_size, device):
            fg = batch['fg'].to(device).to(torch.float32) / 127.5 - 1
            ibg = batch['ibg'].to(device).to(torch.float32) / 127.5 - 1
            z = torch.randn([batch_size, model.z_dims], device=device)
            gen_img, rec_img = model(fg, ibg, z)
            gen_img = (gen_img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            rec_img = (rec_img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return img
        return eval_fn


    def get_sample_fn(self):
        def sample_fn(model, _batch, batch_size, device):
            fg = batch['fg'].to(device).to(torch.float32) / 127.5 - 1
            ibg = batch['ibg'].to(device).to(torch.float32) / 127.5 - 1
            z = torch.randn([batch_size, model.z_dims], device=device)
            gen_img, rec_img = model(fg, ibg, z)
            gen_img = (gen_img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            rec_img = (rec_img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return img
        return sample_fn


    def _init_modules(self):
        cfg = self.cfg

        # TODO: initialized aae should be considered not from scratch
        from_scratch = not cfg.resume

        # build models
        print_(self.rank, '[INFO] initializing models...')
        self.G = build_model(
            cfg,
            cfg.model.G,
        ).train().requires_grad_(False).to(self.device)
        self.D = build_model(
            cfg,
            cfg.model.D,
        ).train().requires_grad_(False).to(self.device)

        # build loss models
        self.face_loss_model = FaceLoss().eval().requires_grad_(False).to(
            self.device)
        self.lpips_loss_model = LpipsLoss().eval().requires_grad_(False).to(
            self.device)

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

        # generator phases
        assert cfg.loss.G.ppl.freq > 1
        assert cfg.loss.G.rec.freq == 1, 'TODO'
        G_opt = get_optimizer(
            cfg,
            getattr(cfg.opt, 'G'),
            module.parameters(),
            reg_interval=cfg.loss.G.ppl.freq,
        )
        self.phases.append(TrainingPhase(
            name='G_gen_main',
            module=self.G,
            opt=G_opt,
            interval=1,
            device=self.device,
            rank=self.rank,
        ))
        self.phases.append(TrainingPhase(
            name='G_gen_reg',
            module=self.G,
            opt=G_opt,
            interval=cfg.loss.G.ppl.freq,
            device=self.device,
            rank=self.rank,
        ))
        self.phases.append(TrainingPhase(
            name='G_gen_reg',
            module=self.G,
            opt=G_opt,
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
            ('G_f', self.G.f, True),
            ('G_g', self.G.g, True),
            ('G_e', self.G.e, True),
            ('D', self.D, True),
            (None, self.G_ema, False),
            ('augment_pipe', self.augment_pipe, True),
        ]


    def print_models(self):
        fg = torch.empty(
            [self.batch_gpu, 3, self.G.imsize, self.G.imsize],
            device=self.device,
        )
        ibg = torch.empty(
            [self.batch_gpu, 3, self.G.imsize, self.G.imsize],
            device=self.device,
        )
        z = torch.empty(
            [self.batch_gpu, self.G.z_dims],
            device=self.device,
        )
        gen_img, rec_img = print_module_summary(self.G, [fg, ibg, z])
        print_module_summary(self.D, [gen_img])
