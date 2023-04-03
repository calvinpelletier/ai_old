#!/usr/bin/env python3
import torch
import torch.nn as nn


pos_encoder = PositionalEncoding(512)
encoder_layers = nn.TransformerEncoderLayer(512, 4, dim_feedforward=1024)
transformer_encoder = nn.TransformerEncoder(encoder_layers, 4)
src = torch.rand(64, 32, 512)
out = transformer_encoder(src)
