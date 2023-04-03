#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Sequence
from itertools import chain
from torchvision import models
from ai_old.loss.base import BaseLoss


class LpipsSoloLoss(BaseLoss):
    def __init__(self, loss_conf):
        super().__init__()
        self.gen_key = loss_conf.gen_key
        self.target_key = loss_conf.target_key
        self.loss = LpipsLoss()

    def forward(self, ents):
        gen = ents[self.gen_key]
        target = ents[self.target_key]

        # loss, sublosses, subloss_times
        return self.loss(gen, target), None, None



class LpipsLoss(nn.Module):
    def __init__(self, type='alex'):
        super().__init__()
        version = '0.1'
        self.net = get_network(type).to('cuda')
        self.lin = LinLayers(self.net.n_channels_list).to('cuda')
        self.lin.load_state_dict(get_state_dict(type, version))

    def forward(self, x, y):
        x = F.interpolate(
            x,
            size=(224, 224),
            mode='bilinear',
            align_corners=False,
        )
        y = F.interpolate(
            y,
            size=(224, 224),
            mode='bilinear',
            align_corners=False,
        )
        feat_x, feat_y = self.net(x), self.net(y)
        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]
        return torch.sum(torch.cat(res, 0)) / x.shape[0]


def get_network(type):
    if type == 'alex':
        return AlexNet()
    elif type == 'squeeze':
        return SqueezeNet()
    elif type == 'vgg':
        return VGG16()
    else:
        raise Exception('unknown lpips model type')


class LinLayers(nn.ModuleList):
    def __init__(self, n_channels_list):
        super().__init__([
            nn.Sequential(
                nn.Identity(),
                nn.Conv2d(nc, 1, 1, 1, 0, bias=False)
            ) for nc in n_channels_list
        ])
        for p in self.parameters():
            p.requires_grad = False


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        # register buffer
        self.register_buffer(
            'mean', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer(
            'std', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def set_requires_grad(self, state: bool):
        for param in chain(self.parameters(), self.buffers()):
            param.requires_grad = state

    def z_score(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor):
        x = self.z_score(x)

        output = []
        for i, (_, layer) in enumerate(self.layers._modules.items(), 1):
            x = layer(x)
            if i in self.target_layers:
                output.append(normalize_activation(x))
            if len(output) == len(self.target_layers):
                break
        return output


class SqueezeNet(BaseNet):
    def __init__(self):
        super(SqueezeNet, self).__init__()

        self.layers = models.squeezenet1_1(True).features
        self.target_layers = [2, 5, 8, 10, 11, 12, 13]
        self.n_channels_list = [64, 128, 256, 384, 384, 512, 512]

        self.set_requires_grad(False)


class AlexNet(BaseNet):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.layers = models.alexnet(True).features
        self.target_layers = [2, 5, 8, 10, 12]
        self.n_channels_list = [64, 192, 384, 256, 256]

        self.set_requires_grad(False)


class VGG16(BaseNet):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layers = models.vgg16(True).features
        self.target_layers = [4, 9, 16, 23, 30]
        self.n_channels_list = [64, 128, 256, 512, 512]

        self.set_requires_grad(False)


def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def get_state_dict(net_type: str = 'alex', version: str = '0.1'):
    # build url
    url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' \
        + f'master/lpips/weights/v{version}/{net_type}.pth'

    # download
    old_state_dict = torch.hub.load_state_dict_from_url(
        url, progress=True,
        map_location=None if torch.cuda.is_available() else torch.device('cpu')
    )

    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val

    return new_state_dict
