#!/usr/bin/env python3
from torch.optim import Adam
from external.optimizer.ranger import Ranger
# from adabelief_pytorch import AdaBelief
# from ranger_adabelief import RangerAdaBelief


def get_optimizer(full_cfg, opt_cfg, params, reg_interval=None):
    if opt_cfg.type == 'ranger':
        return Ranger(params, lr=opt_cfg.lr)

    elif opt_cfg.type == 'adabelief':
        assert opt_cfg.hparams in ['small_gan', 'large_gan']
        return AdaBelief(
            params,
            lr=2e-4,
            eps=1e-16 if opt_cfg.hparams == 'large_gan' else 1e-12,
            betas=(0.5, 0.999),
            weight_decay=0,
            rectify=opt_cfg.hparams == 'large_gan',
            fixed_decay=False,
            amsgrad=False,
        )

    elif opt_cfg.type == 'ranger_adabelief':
        assert opt_cfg.hparams in ['small_gan', 'large_gan']
        return RangerAdaBelief(
            params,
            lr=2e-4,
            eps=1e-16 if opt_cfg.hparams == 'large_gan' else 1e-12,
            betas=(0.5, 0.999),
            weight_decay=0,
        )

    elif opt_cfg.type == 'adam':
        assert opt_cfg.hparams == 'auto-sg2'
        lr = 0.002 if full_cfg.dataset.imsize >= 1024 else 0.0025
        betas = [0, 0.99]
        if reg_interval is not None:
            mb_ratio = reg_interval / (reg_interval + 1)
            lr *= mb_ratio
            betas = [beta ** mb_ratio for beta in betas]
        return Adam(
            params,
            lr=lr,
            betas=betas,
            eps=1e-8,
        )

    else:
        raise Exception('unknown optimizer type: ' + opt_cfg.type)
