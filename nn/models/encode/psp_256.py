#!/usr/bin/env python3
import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from external.op import bias_act


class EqualLinear(nn.Module):
    def __init__(self,
        in_features,
        out_features,
        bias=True,
        activation='linear',
        lr_multiplier=1,
        bias_init=0,
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(
            torch.randn([out_features, in_features]) / lr_multiplier,
        )
        self.bias = torch.nn.Parameter(
            torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x


class GradualStyleBlock(nn.Module):
    def __init__(self, in_c, out_c, spatial):
        super().__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
	norm = torch.norm(input, 2, axis, True)
	output = torch.div(input, norm)
	return output


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
	""" A named tuple describing a ResNet block. """


def get_block(in_channel, depth, num_units, stride=2):
	return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
	if num_layers == 50:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=4),
			get_block(in_channel=128, depth=256, num_units=14),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	elif num_layers == 100:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=13),
			get_block(in_channel=128, depth=256, num_units=30),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	elif num_layers == 152:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=8),
			get_block(in_channel=128, depth=256, num_units=36),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	else:
		raise ValueError("Invalid number of layers: {}. Must be one of [50, 100, 152]".format(num_layers))
	return blocks


class SEModule(nn.Module):
	def __init__(self, channels, reduction):
		super().__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		module_input = x
		x = self.avg_pool(x)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.sigmoid(x)
		return module_input * x


class bottleneck_IR(nn.Module):
	def __init__(self, in_channel, depth, stride):
		super().__init__()
		if in_channel == depth:
			self.shortcut_layer = nn.MaxPool2d(1, stride)
		else:
			self.shortcut_layer = nn.Sequential(
				nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
				nn.BatchNorm2d(depth)
			)
		self.res_layer = nn.Sequential(
			nn.BatchNorm2d(in_channel),
			nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), nn.PReLU(depth),
			nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False), nn.BatchNorm2d(depth)
		)

	def forward(self, x):
		shortcut = self.shortcut_layer(x)
		res = self.res_layer(x)
		return res + shortcut


class bottleneck_IR_SE(nn.Module):
	def __init__(self, in_channel, depth, stride):
		super().__init__()
		if in_channel == depth:
			self.shortcut_layer = nn.MaxPool2d(1, stride)
		else:
			self.shortcut_layer = nn.Sequential(
				nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
				nn.BatchNorm2d(depth)
			)
		self.res_layer = nn.Sequential(
			nn.BatchNorm2d(in_channel),
			nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
			nn.PReLU(depth),
			nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
			nn.BatchNorm2d(depth),
			SEModule(depth, 16)
		)

	def forward(self, x):
		shortcut = self.shortcut_layer(x)
		res = self.res_layer(x)
		return res + shortcut


class Psp256Encoder(nn.Module):
    def __init__(self, w_avg, num_layers=50, mode='ir', n_styles=18, opts=None):
        super().__init__()

        self.register_buffer('w_avg', w_avg)

        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE

        self.input_layer = nn.Sequential(nn.Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      nn.BatchNorm2d(64),
                                      nn.PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = nn.Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = n_styles
        for i in range(self.style_count):
            style = GradualStyleBlock(512, 512, 16)
            self.styles.append(style)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        latents = []
        for j in range(self.style_count):
            latents.append(self.styles[j](x) + self.w_avg)
        out = torch.stack(latents, dim=1)
        return out
