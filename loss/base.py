#!/usr/bin/env python3
import torch.nn as nn


class BaseLoss(nn.Module):
    def forward(self, ents):
        raise NotImplementedError('BaseLoss.forward')

    def has_multiple_losses(self):
        return False

    def get_loss_names(self):
        raise NotImplementedError(
            'need to implement get_loss_names if has_multiple_losses() == True')

    def requires_grad_for_y(self):
        return False
