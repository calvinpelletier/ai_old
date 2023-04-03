#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_old.nn.blocks.conv import ConvBlock
from ai_old.nn.blocks.se import SEModule
from ai_old.util.params import init_params
from ai_old.nn.blocks.arcface import bottleneck_IR_SE
from ai_old.nn.blocks.etc import Flatten
from ai_old.nn.blocks.linear import LinearBlock


class _Block(nn.Module):
    def __init__(self, nc=64, norm='batch', actv='relu'):
        super().__init__()

        self.conv1 = ConvBlock(nc, nc, norm=norm, actv=actv)
        self.conv2 = ConvBlock(nc, nc, norm=norm, actv='none')
        self.se = SEModule(nc, 16)
        self.final = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        return self.final(x + res)


class NN(nn.Module):
    def __init__(self,
        input_state_depth,
        grid_action_depth,
        action_size,
        board_size=8,
        nc=64,
        n_blocks=8,
        norm='batch',
        actv='relu',
        block_type='maia',
        norm_first_layer=True,
        value_head_nc=8,
    ):
        super().__init__()
        self.action_size = action_size

        # body
        blocks = [ConvBlock(
            input_state_depth,
            nc,
            norm=norm if norm_first_layer else 'none',
            actv=actv,
        )]
        for i in range(n_blocks):
            if block_type == 'maia':
                blocks.append(_Block(nc, norm=norm, actv=actv))
            elif block_type == 'ir_se':
                blocks.append(bottleneck_IR_SE(nc, nc, 1))
            else:
                raise Exception(block_type)

        self.body = nn.Sequential(*blocks)

        # value head
        self.value_head = nn.Sequential(
            ConvBlock(
                nc,
                nc,
                norm=norm,
                actv=actv,
            ),
            ConvBlock(
                nc,
                value_head_nc,
                norm=norm,
                actv=actv,
            ),
            Flatten(),
            nn.Linear(board_size * board_size * value_head_nc, 1),
            nn.Tanh(),
        )

        # policy head
        self.policy_head = nn.Sequential(
            ConvBlock(
                nc,
                nc,
                norm=norm,
                actv=actv,
            ),
            ConvBlock(
                nc,
                grid_action_depth,
                norm='none',
                actv='none',
            ),
        )

        self.apply(init_params())

    def forward(self, input_state):
        enc = self.body(input_state)
        # policy = F.log_softmax(torch.reshape(
        #     self.policy_head(enc),
        #     (input_state.shape[0], self.action_size),
        # ), dim=1)
        policy = torch.reshape(
            self.policy_head(enc),
            (input_state.shape[0], self.action_size),
        )
        value = self.value_head(enc).squeeze(1)
        return policy, value


class NN2(nn.Module):
    def __init__(self,
        state_depth,
        ctx_depth,
        meta_dims,
        grid_action_depth,
        action_size,
        board_size=8,
    ):
        super().__init__()

        # state
        self.state_encoder = nn.Sequential(
            ConvBlock(state_depth, 128, k=1, norm='none'),
            bottleneck_IR_SE(128, 128, 1),
            bottleneck_IR_SE(128, 128, 1),
            bottleneck_IR_SE(128, 128, 1),
            bottleneck_IR_SE(128, 128, 1),
        )
        self.state_weight = nn.Parameter(torch.tensor(1.))

        # ctx
        self.ctx_encoder = nn.Sequential(
            ConvBlock(ctx_depth, 256, k=1, norm='none'),
            bottleneck_IR_SE(256, 128, 1),
            bottleneck_IR_SE(128, 128, 1),
        )
        self.ctx_weight = nn.Parameter(torch.tensor(1.))

        # meta
        self.meta_encoder = nn.Sequential(
            LinearBlock(meta_dims, 16, norm='none'),
            LinearBlock(16, 32),
            LinearBlock(32, 128),
        )
        self.meta_weight = nn.Parameter(torch.tensor(1.))

        # main
        self.main_encoder = nn.Sequential(
            bottleneck_IR_SE(128, 64, 1),
            bottleneck_IR_SE(64, 64, 1),
            bottleneck_IR_SE(64, 64, 1),
            bottleneck_IR_SE(64, 64, 1),
        )

        # value
        self.value_head = nn.Sequential(
            bottleneck_IR_SE(64, 64, 1),
            ConvBlock(64, 8, k=1),
            Flatten(),
            nn.Linear(8 * board_size * board_size, 1),
            nn.Tanh(),
        )

        # policy
        self.policy_head = nn.Sequential(
            bottleneck_IR_SE(64, 64, 1),
            bottleneck_IR_SE(64, 64, 1),
        )
        self.policy_head_spatial = ConvBlock(
            64,
            grid_action_depth,
            k=1,
            norm='none',
            actv='none',
        )
        n_universal_actions = action_size - (grid_action_depth * 8 * 8)
        assert n_universal_actions == 1
        self.policy_head_universal = nn.Sequential(
            ConvBlock(64, n_universal_actions, k=1),
            Flatten(),
            nn.Linear(
                n_universal_actions * board_size * board_size,
                n_universal_actions,
            ),
        )

        self.apply(init_params())

    def forward(self, state, ctx, meta):
        bs = state.shape[0]

        state_enc = self.state_encoder(state) * self.state_weight
        ctx_enc = self.ctx_encoder(ctx) * self.ctx_weight
        meta_enc = self.meta_encoder(meta) * self.meta_weight
        meta_enc = meta_enc.reshape(bs, 128, 1, 1).repeat(1, 1, 8, 8)
        combo = state_enc + ctx_enc + meta_enc
        enc = self.main_encoder(combo)

        policy_enc = self.policy_head(enc)
        spatial_policy = torch.reshape(
            self.policy_head_spatial(policy_enc),
            (bs, -1),
        )
        universal_policy = self.policy_head_universal(policy_enc)
        policy = torch.cat([spatial_policy, universal_policy], dim=1)

        value = self.value_head(enc).squeeze(1)

        return policy, value
