#!/usr/bin/env python3
from ai_old.util.etc import print_
from external.optimizer import get_optimizer
from ai_old.trainer.phase import TrainingPhase
from external.sg2 import training_stats
from ai_old.util.factory import build_loss
import torch.nn as nn


class BaseTrainer:
    def __init__(self, cfg, rank, device):
        self.cfg = cfg
        self.rank = rank
        self.device = device
        self.num_gpus = cfg.num_gpus
        self.batch_size = cfg.dataset.batch_size
        self.batch_gpu = cfg.dataset.batch_gpu

        self._init_modules()
        self._distribute_modules()
        self._init_loss()
        self._init_phases()


    def _init_modules(self):
        raise NotImplementedError('_init_modules')


    def _init_phases(self):
        raise NotImplementedError('_get_phases')


    def _get_modules_for_distribution(self):
        raise NotImplementedError('_get_modules_for_distribution')


    def run_batch(self, batch, batch_idx, cur_step):
        raise NotImplementedError('run_batch')


    def get_modules_for_save(self):
        raise NotImplementedError('get_modules_for_save')


    def print_models(self):
        raise NotImplementedError('print_models')


    def collect_phase_stats(self):
        for phase in self.phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('timing/' + phase.name, value)


    def _init_loss(self):
        print_(self.rank, '[INFO] initializing loss...')
        self.loss = build_loss(self.cfg, self, self.device)


    def _distribute_modules(self):
        print_(self.rank, f'[INFO] distributing across {self.num_gpus} gpus...')
        self.ddp_modules = dict()
        for name, module, req_grad in self._get_modules_for_distribution():
            if (self.num_gpus > 1) and (module is not None) and \
                    len(list(module.parameters())) != 0 and req_grad:
                nn.SyncBatchNorm.convert_sync_batchnorm(module)
                module.requires_grad_(True)
                module = nn.parallel.DistributedDataParallel(
                    module,
                    device_ids=[self.device],
                    broadcast_buffers=False,
                )
                module.requires_grad_(False)
            if name is not None:
                self.ddp_modules[name] = module


    def _build_loss_phase(self, name, module):
        cfg = self.cfg
        opt = get_optimizer(
            cfg,
            cfg.opt,
            module.parameters(),
            reg_interval=None,
        )
        return TrainingPhase(
            name=name,
            module=module,
            opt=opt,
            interval=1,
            device=self.device,
            rank=self.rank,
        )


    def _build_regularized_loss_phases(self, name, module, reg_interval):
        cfg = self.cfg
        phases = []
        assert reg_interval >= 1
        opt = get_optimizer(
            cfg,
            getattr(cfg.opt, name),
            module.parameters(),
            reg_interval=reg_interval,
        )
        if reg_interval == 1:
            phases.append(TrainingPhase(
                name=name + 'both',
                module=module,
                opt=opt,
                interval=1,
                device=self.device,
                rank=self.rank,
            ))
        else:
            phases.append(TrainingPhase(
                name=name + 'main',
                module=module,
                opt=opt,
                interval=1,
                device=self.device,
                rank=self.rank,
            ))
            phases.append(TrainingPhase(
                name=name + 'reg',
                module=module,
                opt=opt,
                interval=reg_interval,
                device=self.device,
                rank=self.rank,
            ))
        return phases
