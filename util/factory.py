#!/usr/bin/env python3
import torch
import importlib
from ai_old.util import config
from ai_old.util.params import requires_grad
import ai_old.constants as c
import os
from copy import deepcopy
from ai_old.server.common import log, cloud_storage_client
from external.sg2.augment import AugmentPipe
import pickle


def build_model(full_cfg, model_cfg, verbose=True):
    cls = convert_importpath(model_cfg.type)

    # NOTE: dont delete type from the og config because we might need to reuse
    # (like for model ema)
    model_cfg_dict = deepcopy(vars(model_cfg))
    del model_cfg_dict['type']

    model = cls(full_cfg, **model_cfg_dict)

    return model


def build_model_from_exp(exp, model_label, return_cfg=True, migrate_to_pt=True):
    cfg = config.load(os.path.join(
        c.CONFIGS_FOLDER,
        'train',
        exp + '.yaml',
    ))

    dir = os.path.join(c.EXP_PATH, exp, 'saves')
    pt_model_path = os.path.join(dir, 'latest.pt')
    pkl_model_path = os.path.join(dir, 'latest.pkl')

    if os.path.isfile(pt_model_path):
        if model_label == 'model':
            model = build_model(cfg, cfg.model)
        elif model_label == 'G' or model_label == 'G_ema':
            model = build_model(cfg, cfg.model.G)
        elif model_label == 'D':
            model = build_model(cfg, cfg.model.D)
        else:
            raise Exception('todo')
        model.load_state_dict(torch.load(pt_model_path)[model_label])
    else:
        with open(pkl_model_path, 'rb') as f:
            model = pickle.load(f)[model_label]

        if migrate_to_pt:
            # switch to pt format for future loads
            torch.save({model_label: model.state_dict()}, pt_model_path)

    if return_cfg:
        return model, cfg
    return model


def legacy_build_model(model_cfg, verbose=True):
    cls = convert_importpath(model_cfg.type)

    # NOTE: dont delete type from the og config because we might need to reuse
    # (like for model ema)
    model_cfg_dict = deepcopy(vars(model_cfg))
    del model_cfg_dict['type']

    model = cls(**model_cfg_dict)

    return model


def legacy_build_model_from_exp(exp, freeze=True, verbose=True):
    print(f'[INFO] building model from exp: {exp}')
    exp_conf = config.load(os.path.join(
        c.CONFIGS_FOLDER,
        'train',
        exp + '.yaml',
    ))
    model = legacy_build_model(exp_conf.model, verbose=verbose)

    latest_path = os.path.join(c.EXP_PATH, exp, 'saves', 'latest.pt')
    if not os.path.exists(latest_path):
        log().info("Missing model weights at %s, downloading from GCS..." % latest_path)
        cloud_path = os.path.join("model-data/exp", exp, "latest.pt")
        cloud_storage_client.get_client().download_file(cloud_path, latest_path)

    model.load(torch.load(latest_path)['model'])
    if freeze:
        requires_grad(model, False)
    return model


def build_trainer(full_cfg, rank, device):
    cls = convert_importpath(full_cfg.trainer.type)
    return cls(full_cfg, rank, device)


# def build_subtrainer(wrapper, conf, gpus, base_results_path, n_epochs):
#     sub_results_path = os.path.join(base_results_path, conf.name)
#     cls = convert_importpath(conf.type)
#     del conf.type
#     return cls(wrapper, conf, gpus, sub_results_path, n_epochs)


def build_loss(full_cfg, trainer, device):
    cls = convert_importpath(full_cfg.loss.type)
    return cls(full_cfg, trainer, device)


def build_dataset(conf):
    cls = convert_importpath(conf.type)
    xflip = hasattr(conf, 'xflip') and conf.xflip
    return cls(xflip)


def build_task(full_cfg, rank, device):
    cls = convert_importpath(full_cfg.task)
    return cls(full_cfg, rank, device)


def build_augpipe(aug_cfg):
    augpipe_kwargs = {
        'blit':   dict(xflip=1, rotate90=1, xint=1),
        'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color':  dict(
            brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1,
        ),
        'filter': dict(imgfilter=1),
        'noise':  dict(noise=1),
        'cutout': dict(cutout=1),
        'bg':     dict(
            xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1,
        ),
        'bgc':    dict(
            xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1,
            brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1,
        ),
        'bgcf':   dict(
            xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1,
            brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1,
            imgfilter=1,
        ),
        'bgcfn':  dict(
            xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1,
            brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1,
            imgfilter=1, noise=1,
        ),
        'bgcfnc': dict(
            xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1,
            brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1,
            imgfilter=1, noise=1, cutout=1,
        ),
    }
    return AugmentPipe(**augpipe_kwargs[aug_cfg.augpipe])


def convert_importpath(importpath):
    parts = importpath.split('.')
    module = '.'.join(parts[:-1])
    class_name = parts[-1]
    return getattr(importlib.import_module(module), class_name)
