#!/usr/bin/env python3
from external.sg2 import persistence
import torch.nn as nn
from external.sg2.unit import FullyConnectedLayer


@persistence.persistent_class
class GenderLatentTransform(nn.Module):
    def __init__(self, z_dims=512, n_mlp_layers=2):
        super().__init__()
        self.encode_gender = FullyConnectedLayer(1, z_dims, activation='linear')
        self.net = nn.Sequential(*[
            FullyConnectedLayer(
                z_dims,
                z_dims,
                activation='linear' if j == n_mlp_layers - 1 else 'lrelu',
            )
            for j in range(n_mlp_layers)
        ])

    def forward(self, z, src_gender):
        # print('z', z.shape)
        # print('src_gender', src_gender.shape)
        g_enc = self.encode_gender(src_gender)
        gendered_z = z + g_enc
        return self.net(gendered_z)


# class GenderLatentTransform(Unit):
#     def __init__(self, z_dims=512):
#         super().__init__()
#         self.encode_gender = nn.Linear(1, z_dims)
#         self.net = nn.Linear(z_dims, z_dims)
#
#     def forward(self, z, src_gender):
#         g_enc = self.encode_gender(src_gender)
#         gendered_z = z + g_enc
#         return self.net(gendered_z)
