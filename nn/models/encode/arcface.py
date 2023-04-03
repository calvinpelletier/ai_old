#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_old.nn.blocks.etc import Flatten
from ai_old.nn.blocks.arcface import *
import ai_old.constants as c
import os
from ai_old.util.params import init_params, requires_grad


class ArcFaceWrapper(nn.Module):
    def __init__(self,
        imsize=112,
        resize_input=True,
        pretrained=True,
        frozen=True,
        unfreeze_at_epoch=None,
        num_ws=None,
    ):
        super().__init__()
        self.imsize = imsize
        self.resize_input = resize_input
        self.pretrained = pretrained
        self.frozen = frozen
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.num_ws = num_ws

        # both not pretrained and frozen would be weird
        assert pretrained or not frozen

        self.model = ArcFace(
            input_size=self.imsize,
            num_layers=50,
            drop_ratio=0.6,
            mode='ir_se',
        )
        self.dims_out = 512

        if pretrained:
            self.model.load_state_dict(torch.load(os.path.join(
                c.PRETRAINED_MODELS,
                'arcface/model_ir_se50.pth',
            )))

        if frozen:
            requires_grad(self.model, False)

    def forward(self, x):
        if self.resize_input:
            x = F.interpolate(
                x,
                size=(self.imsize, self.imsize),
                mode='bilinear',
                align_corners=True,
            )
        z = self.model(x)
        if self.num_ws is None:
            return z
        return z.unsqueeze(1).repeat([1, self.num_ws, 1])

    # def end_of_epoch(self, epoch):
    #     if self.frozen and self.unfreeze_at_epoch is not None and \
    #             epoch >= self.unfreeze_at_epoch:
    #         self.frozen = False
    #         requires_grad(self.model, True)
    #
    # def init_params(self):
    #     if not self.pretrained:
    #         self.apply(init_params())


class ArcFace(nn.Module):
    def __init__(self,
        input_size,
        num_layers,
        mode='ir',
        drop_ratio=0.4,
        affine=True,
    ):
        super().__init__()
        assert input_size in [112, 224]
        assert num_layers in [50, 100, 152]
        assert mode in ['ir', 'ir_se']
        blocks = get_blocks(num_layers)
        if mode == 'ir':
        	unit_module = bottleneck_IR
        elif mode == 'ir_se':
        	unit_module = bottleneck_IR_SE
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
        	nn.BatchNorm2d(64),
        	nn.PReLU(64),
        )
        if input_size == 112:
        	self.output_layer = nn.Sequential(
                nn.BatchNorm2d(512),
        	    nn.Dropout(drop_ratio),
        	    Flatten(),
        	    nn.Linear(512 * 7 * 7, 512),
        	    nn.BatchNorm1d(512, affine=affine,
            ))
        else:
        	self.output_layer = nn.Sequential(
                nn.BatchNorm2d(512),
                nn.Dropout(drop_ratio),
                Flatten(),
                nn.Linear(512 * 14 * 14, 512),
                nn.BatchNorm1d(512, affine=affine),
            )

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(
                    bottleneck.in_channel,
        			bottleneck.depth,
        			bottleneck.stride,
                ))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        out = torch.div(x, norm)
        return out
