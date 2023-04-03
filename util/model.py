#!/usr/bin/env python3
import torch.nn as nn


def accumulate(avg, model, decay=0.999):
    if isinstance(model, nn.DataParallel):
        model = model.module
    avg_params = dict(avg.named_parameters())
    model_params = dict(model.named_parameters())

    for k in avg_params.keys():
        avg_params[k].data.mul_(decay).add_(model_params[k].data, alpha=1 - decay)
