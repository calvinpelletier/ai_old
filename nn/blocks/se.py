#!/usr/bin/env python3
import torch.nn as nn
from external.sg2 import persistence
from ai_old.nn.blocks.conv import ConvBlock, CustomConvBlock


@persistence.persistent_class
class SEModule(nn.Module):
	def __init__(self, nc, reduction, conv_clamp=None):
		super().__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		if conv_clamp is not None:
			self.conv0 = CustomConvBlock(
	            nc,
	            nc // reduction,
	            k=1,
	            s=1,
	            norm='none',
	            actv='relu',
	            pad=0,
	            bias=False,
				conv_clamp=conv_clamp,
	        )
			self.conv1 = CustomConvBlock(
	            nc // reduction,
	            nc,
	            k=1,
	            s=1,
	            norm='none',
	            actv='sigmoid',
	            pad=0,
	            bias=False,
				conv_clamp=conv_clamp,
	        )
		else:
			self.conv0 = ConvBlock(
	            nc,
	            nc // reduction,
	            k=1,
	            s=1,
	            norm='none',
	            actv='relu',
	            pad=0,
	            bias=False,
	        )
			self.conv1 = ConvBlock(
	            nc // reduction,
	            nc,
	            k=1,
	            s=1,
	            norm='none',
	            actv='sigmoid',
	            pad=0,
	            bias=False,
	        )

	def forward(self, x):
		module_input = x
		x = self.avg_pool(x)
		x = self.conv0(x)
		x = self.conv1(x)
		return module_input * x
