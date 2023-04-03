#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from ai_old.util.etc import print_
from ai_old.trainer.base import BaseTrainer
from external.sg2.misc import print_module_summary
from ai_old.util.factory import build_model
import external.sg2.misc as misc
from ai_old.loss.perceptual.face import FaceIdLoss
from ai_old.loss.clip import GenderSwapClipLoss, GenderSwapClipDirLoss
from ai_old.nn.models.seg.seg import colorize


class SegFromGenTrainer(BaseTrainer):
    def run_batch(self, batch, batch_idx, cur_step):
        # prep_data
        with torch.autograd.profiler.record_function('data_prep'):
            phase_w = batch['w']
            phase_w = (
                phase_w.to(self.device).to(torch.float32)
            ).split(self.batch_gpu)

            phase_seg = batch['seg']
            phase_seg = (
                phase_seg.to(self.device).to(torch.long)
            ).split(self.batch_gpu)

        # run training phases
        for phase in self.phases:
            if batch_idx % phase.interval != 0:
                continue

            phase.init_gradient_accumulation()

            # accumulate gradients over multiple rounds
            for round_idx, (w, seg) in enumerate(zip(phase_w, phase_seg)):
                sync = (round_idx == self.batch_size // \
                    (self.batch_gpu * self.num_gpus) - 1)
                gain = phase.interval
                self.loss.accumulate_gradients(
                    phase=phase.name,
                    w=w,
                    seg=seg,
                    sync=sync,
                    gain=gain,
                )

            phase.update_params()


    def run_G(self, w, sync):
        G_g = self.ddp_modules['G_g']
        G_s = self.ddp_modules['G_s']

        with misc.ddp_sync(G_g, sync):
            _, feats = G_g(w)

        with misc.ddp_sync(G_s, sync):
            seg = G_s(feats)

        return seg


    def get_eval_fn(self):
        def _eval_fn(G, batch, batch_size, device):
            w = batch['w'].to(device).to(torch.float32)
            gt = batch['seg'].to(device).to(torch.long)
            _, feats = G.g(w)
            pred = G.s(feats)
            return gt, pred
        return _eval_fn


    def get_sample_fn(self):
        def _sample_fn(model, batch, batch_size, device):
            w = batch['w'].to(device).to(torch.float32)
            _, feats = model.g(w)
            pred = model.s(feats)
            return colorize(pred)
        return _sample_fn


    def _init_modules(self):
        cfg = self.cfg

        # build main model
        print_(self.rank, '[INFO] initializing model...')
        self.G = build_model(
            cfg,
            cfg.model,
        ).train().requires_grad_(False).to(self.device)

        # resume training
        assert not cfg.resume


    def _init_phases(self):
        print_(self.rank, '[INFO] initializing training phases...')
        self.phases = [self._build_loss_phase('seg', self.G)]


    def get_modules_for_save(self):
        return [('G', self.G)]


    def _get_modules_for_distribution(self):
        return [
            ('G_g', self.G.g, False),
            ('G_s', self.G.s, True),
        ]


    def print_models(self):
        feats = [
            torch.empty([self.batch_gpu, 512, 32, 32], device=self.device),
            torch.empty([self.batch_gpu, 512, 64, 64], device=self.device),
            torch.empty([self.batch_gpu, 256, 128, 128], device=self.device),
            torch.empty([self.batch_gpu, 128, 256, 256], device=self.device),
            torch.empty([self.batch_gpu, 64, 512, 512], device=self.device),
            torch.empty([self.batch_gpu, 32, 1024, 1024], device=self.device),
        ]
        _ = print_module_summary(self.G.s, [feats])
