#!/usr/bin/env python3
import torch
from ai_old.util.etc import print_
from ai_old.trainer.base import BaseTrainer
from external.sg2.misc import print_module_summary
from ai_old.util.factory import build_model
import external.sg2.misc as misc
from ai_old.nn.models.seg.seg import colorize_seg


class OuterSegTrainer(BaseTrainer):
    def run_batch(self, batch, batch_idx, cur_step):
        # prep_data
        with torch.autograd.profiler.record_function('data_prep'):
            phase_seg = batch['seg']
            phase_seg = (
                phase_seg.to(self.device).to(torch.long)
            ).split(self.batch_gpu)

            phase_img = batch['img']
            phase_img = (
                phase_img.to(self.device).to(torch.float32) / 127.5 - 1
            ).split(self.batch_gpu)

        # run training phases
        for phase in self.phases:
            if batch_idx % phase.interval != 0:
                continue

            phase.init_gradient_accumulation()

            # accumulate gradients over multiple rounds
            for round_idx, (seg, img) in enumerate(zip(phase_seg, phase_img)):
                sync = (round_idx == self.batch_size // \
                    (self.batch_gpu * self.num_gpus) - 1)
                gain = phase.interval
                self.loss.accumulate_gradients(
                    phase=phase.name,
                    seg=seg,
                    img=img,
                    sync=sync,
                    gain=gain,
                )

            phase.update_params()


    def run_model(self, seg, img, sync):
        model = self.ddp_modules['model']

        with misc.ddp_sync(model, sync):
            out = model(seg, img)

        return out


    def get_eval_fn(self):
        def _eval_fn(model, batch, batch_size, device):
            seg = batch['seg'].to(device).to(torch.long)
            img = batch['img'].to(device).to(torch.float32) / 127.5 - 1
            out = model(seg, img)
            return seg, out
        return _eval_fn


    def get_sample_fn(self):
        def _sample_fn(model, batch, batch_size, device):
            seg = batch['seg'].to(device).to(torch.long)
            img = batch['img'].to(device).to(torch.float32) / 127.5 - 1
            out = model(seg, img)
            return colorize_seg(out)
        return _sample_fn


    def _init_modules(self):
        cfg = self.cfg

        # build main model
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
        return [
            ('model', self.model, True),
        ]


    def print_models(self):
        imsize = self.cfg.dataset.imsize
        seg = torch.empty(
            [self.batch_gpu, imsize, imsize],
            dtype=torch.long,
            device=self.device,
        )
        img = torch.empty(
            [self.batch_gpu, 3, imsize, imsize],
            device=self.device,
        )
        _ = print_module_summary(self.model, [seg, img])
