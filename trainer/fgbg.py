#!/usr/bin/env python3
import torch
from ai_old.util.etc import print_
from ai_old.trainer.base import BaseTrainer
from external.sg2.misc import print_module_summary
from ai_old.util.factory import build_model, build_augpipe
import copy
from external.sg2 import training_stats
import external.sg2.misc as misc
import numpy as np
import external.sg2.legacy as legacy
import os


class FgBgStyleGanTrainer(BaseTrainer):
    def run_batch(self, batch, batch_idx, cur_step):
        # prep_data
        with torch.autograd.profiler.record_function('data_prep'):
            phase_real_img = batch['y']
            phase_real_img = (
                phase_real_img.to(self.device).to(torch.float32) / 127.5 - 1
            ).split(self.batch_gpu)

            phase_real_seg = batch['seg']
            phase_real_seg = (
                phase_real_seg.to(self.device).to(torch.float32).unsqueeze(
                    dim=1)
            ).split(self.batch_gpu)

            all_z_fg = torch.randn(
                [len(self.phases) * self.batch_size, self.G.z_dims_fg],
                device=self.device,
            )
            all_z_fg = [
                phase_z_fg.split(self.batch_gpu) \
                for phase_z_fg in all_z_fg.split(self.batch_size)
            ]

            all_z_bg = torch.randn(
                [len(self.phases) * self.batch_size, self.G.z_dims_bg],
                device=self.device,
            )
            all_z_bg = [
                phase_z_bg.split(self.batch_gpu) \
                for phase_z_bg in all_z_bg.split(self.batch_size)
            ]

        # run training phases
        for phase, phase_z_fg, phase_z_bg in zip(
            self.phases,
            all_z_fg,
            all_z_bg,
        ):
            if batch_idx % phase.interval != 0:
                continue

            phase.init_gradient_accumulation()

            # accumulate gradients over multiple rounds
            for round_idx, (real_img, real_seg, z_fg, z_bg) in enumerate(zip(
                phase_real_img,
                phase_real_seg,
                phase_z_fg,
                phase_z_bg
            )):
                sync = (round_idx == self.batch_size // \
                    (self.batch_gpu * self.num_gpus) - 1)
                gain = phase.interval
                self.loss.accumulate_gradients(
                    phase=phase.name,
                    real_img=real_img,
                    real_seg=real_seg,
                    z_fg=z_fg,
                    z_bg=z_bg,
                    sync=sync,
                    gain=gain,
                )

            phase.update_params()

        self._update_model_ema(cur_step)

        self._run_ada_heuristic(batch_idx)


    def run_G(self, z_fg, z_bg, sync):
        G_f_fg = self.ddp_modules['G_f_fg']
        G_f_bg = self.ddp_modules['G_f_bg']
        G_g_fg = self.ddp_modules['G_g_fg']
        G_g_bg = self.ddp_modules['G_g_bg']
        style_mix_prob = self.cfg.trainer.style_mix_prob

        with misc.ddp_sync(G_f_fg, sync):
            ws_fg = G_f_fg(z_fg, None)
            if style_mix_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty(
                        [],
                        dtype=torch.int64,
                        device=ws_fg.device,
                    ).random_(1, ws_fg.shape[1])
                    cutoff = torch.where(
                        torch.rand([], device=ws_fg.device) < style_mix_prob,
                        cutoff,
                        torch.full_like(cutoff, ws_fg.shape[1]),
                    )
                    ws_fg[:, cutoff:] = G_f_fg(
                        torch.randn_like(z_fg),
                        None,
                        skip_w_avg_update=True,
                    )[:, cutoff:]

        with misc.ddp_sync(G_f_bg, sync):
            ws_bg = G_f_bg(z_bg, None)

        with misc.ddp_sync(G_g_bg, sync):
            x_bg = G_g_bg(ws_bg)

        with misc.ddp_sync(G_g_fg, sync):
            img, seg = G_g_fg(x_bg, ws_fg)

        return img, ws_fg, seg


    def run_D(self, img, seg, sync):
        D = self.ddp_modules['D']
        augment_pipe = self.ddp_modules['augment_pipe']

        # augmentation
        if augment_pipe is not None:
            img = augment_pipe(img)

        # discrimination
        with misc.ddp_sync(D, sync):
            # combo = torch.cat([img, seg], dim=1)
            # logits = D(combo)
            logits = D(img, seg)

        return logits


    def get_eval_fn(self):
        def eval_fn(model, _batch, batch_size, device):
            z_fg = torch.randn([batch_size, model.z_dims_fg], device=device)
            z_bg = torch.randn([batch_size, model.z_dims_bg], device=device)
            img, _ = model(z_fg, z_bg)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return img
        return eval_fn


    def get_sample_fn(self):
        def sample_fn(model, _batch, batch_size, device):
            z_fg1 = torch.randn([batch_size, model.z_dims_fg], device=device)
            z_fg2 = torch.randn([batch_size, model.z_dims_fg], device=device)
            z_bg1 = torch.randn([batch_size, model.z_dims_bg], device=device)
            z_bg2 = torch.randn([batch_size, model.z_dims_bg], device=device)
            img11, seg11 = model(z_fg1, z_bg1)
            img12, seg12 = model(z_fg1, z_bg2)
            img21, seg21 = model(z_fg2, z_bg1)
            img22, seg22 = model(z_fg2, z_bg2)
            # seg11 = (seg11 * 2. - 1.).repeat(1, 3, 1, 1)
            # seg12 = (seg12 * 2. - 1.).repeat(1, 3, 1, 1)
            # seg21 = (seg21 * 2. - 1.).repeat(1, 3, 1, 1)
            # seg22 = (seg22 * 2. - 1.).repeat(1, 3, 1, 1)
            # img = torch.cat([
            #     torch.cat([img11, seg11, img12, seg12], dim=3),
            #     torch.cat([img21, seg21, img22, seg22], dim=3),
            # ], dim=2)
            img = torch.cat([
                torch.cat([img11, img12], dim=3),
                torch.cat([img21, img22], dim=3),
            ], dim=2)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return img
        return sample_fn


    def _update_model_ema(self, cur_step):
        with torch.autograd.profiler.record_function('Gema'):
            ema_steps = self.ema_ksteps * 1000
            if self.ema_rampup is not None:
                ema_steps = min(ema_steps, cur_step * self.ema_rampup)
            ema_beta = 0.5 ** (self.batch_size / max(ema_steps, 1e-8))
            for p_ema, p in zip(self.G_ema.parameters(), self.G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(self.G_ema.buffers(), self.G.buffers()):
                b_ema.copy_(b)


    def _run_ada_heuristic(self, batch_idx):
        cfg = self.cfg
        if (self.ada_stats is not None) and \
                ((batch_idx + 1) % cfg.trainer.aug.freq == 0):
            self.ada_stats.update()
            adjust = np.sign(
                self.ada_stats['Loss/signs/real'] - cfg.trainer.aug.target) * \
                (self.batch_size * cfg.trainer.aug.freq) / \
                (cfg.trainer.aug.speed * 1000)
            self.augment_pipe.p.copy_(
                (self.augment_pipe.p + adjust).max(misc.constant(
                    0,
                    device=self.device,
                )),
            )


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

        # model ema
        self.G_ema = copy.deepcopy(self.G).eval()
        self.ema_ksteps = 10
        self.ema_rampup = 0.05 if from_scratch else None

        # resume training
        if cfg.resume and self.rank == 0:
            latest_path = os.path.join(cfg.saves_dir, 'latest.pkl')
            print(f'[INFO] initializing with {latest_path}...')
            with open(latest_path, 'rb') as f:
                resume_data = legacy.load_network_pkl(f)
            for name, module in [
                ('G', self.G),
                ('D', self.D),
                ('G_ema', self.G_ema),
            ]:
                misc.copy_params_and_buffers(
                    resume_data[name],
                    module,
                    require_all=False,
                )

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
        self.phases += self._build_regularized_loss_phases(
            'G',
            self.G,
            cfg.loss.G.ppl.freq,
        )
        self.phases += self._build_regularized_loss_phases(
            'D',
            self.D,
            cfg.loss.D.gp.freq,
        )


    def get_modules_for_save(self):
        return [
            ('G', self.G),
            ('D', self.D),
            ('G_ema', self.G_ema),
            ('augment_pipe', self.augment_pipe),
        ]


    def _get_modules_for_distribution(self):
        return [
            ('G_f_fg', self.G.f_fg, True),
            ('G_f_bg', self.G.f_bg, True),
            ('G_g_fg', self.G.g_fg, True),
            ('G_g_bg', self.G.g_bg, True),
            ('D', self.D, True),
            (None, self.G_ema, False),
            ('augment_pipe', self.augment_pipe, True),
        ]


    def print_models(self):
        z_fg = torch.empty(
            [self.batch_gpu, self.G.z_dims_fg],
            device=self.device,
        )
        z_bg = torch.empty(
            [self.batch_gpu, self.G.z_dims_bg],
            device=self.device,
        )
        imsize = self.cfg.dataset.imsize
        # seg_img_combo = torch.empty(
        #     [self.batch_gpu, 4, imsize, imsize],
        #     device=self.device,
        # )
        img, seg = print_module_summary(self.G, [z_fg, z_bg])
        print_module_summary(self.D, [img, seg])
