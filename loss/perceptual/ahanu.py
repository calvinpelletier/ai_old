#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_old.nn.models.encode.psp import GradualStyleEncoder
import os
import ai_old.constants as c
import copy


class AhanuPercepLoss(nn.Module):
    def __init__(self, version):
        super().__init__()

        if version == 0:
            self.model = _EncoderV0().cuda().eval()
            self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        elif version == 1:
            self.model = _EncoderV1().cuda().eval()
            self.weights = [1.0 / 32 + 1.0 / 16, 1.0 / 8 + 1.0 / 4 + 1.0]
        elif version == 2:
            self.model = _EncoderV2().cuda().eval()
            self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        else:
            raise Exception(version)

        self.loss = nn.L1Loss()

    def forward(self, x, y):
        x = F.interpolate(
            x,
            size=(256, 256),
            mode='bilinear',
            align_corners=False,
        )
        y = F.interpolate(
            y,
            size=(256, 256),
            mode='bilinear',
            align_corners=False,
        )
        x_out, y_out = self.model(x), self.model(y)
        # ret = torch.cuda.FloatTensor(1).fill_(0)
        ret = self.weights[0] * self.loss(x_out[0], y_out[0])
        for i in range(1, len(x_out)):
            ret += self.weights[i] * self.loss(x_out[i], y_out[i])
        return ret.squeeze()


class _EncoderV0(nn.Module):
    def __init__(self):
        super().__init__()
        psp = GradualStyleEncoder()
        psp.load_state_dict(_get_keys(torch.load(os.path.join(
            c.PRETRAINED_MODELS,
            'psp/psp_ffhq_encode.pt',
        )), 'encoder'), strict=True)

        self.input_layer = copy.deepcopy(psp.input_layer)
        self.body = copy.deepcopy(psp.body)
        del psp

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.input_layer(x)
        s1 = x

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 2:
                s2 = x
            elif i == 6:
                s3 = x
            elif i == 20:
                s4 = x
            elif i == 23:
                s5 = x

        out = [s1, s2, s3, s4, s5]
        return out


class _EncoderV1(nn.Module):
    def __init__(self):
        super().__init__()
        psp = GradualStyleEncoder()
        psp.load_state_dict(_get_keys(torch.load(os.path.join(
            c.PRETRAINED_MODELS,
            'psp/psp_ffhq_encode.pt',
        )), 'encoder'), strict=True)

        self.input_layer = copy.deepcopy(psp.input_layer)
        self.body = copy.deepcopy(psp.body)
        del psp

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        s1 = self.input_layer(x)
        s2 = self.body(s1)
        out = [s1, s2]
        return out


class _EncoderV2(nn.Module):
    def __init__(self):
        super().__init__()
        psp = GradualStyleEncoder()
        psp.load_state_dict(_get_keys(torch.load(os.path.join(
            c.PRETRAINED_MODELS,
            'psp/psp_ffhq_encode.pt',
        )), 'encoder'), strict=True)

        self.input_layer = copy.deepcopy(psp.input_layer)
        self.body = copy.deepcopy(psp.body)
        del psp

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.input_layer(x)
        s1 = x

        modulelist = list(self.body._modules.values())
        for i in range(9):
            x = modulelist[i](x)
            if i == 2:
                s2 = x
            elif i == 4:
                s3 = x
            elif i == 6:
                s4 = x
            elif i == 8:
                s5 = x

        out = [s1, s2, s3, s4, s5]
        return out


def _get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {
        k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name
    }
    return d_filt
