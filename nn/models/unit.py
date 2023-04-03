#!/usr/bin/env python3
import torch
import torch.nn as nn
from ai_old.util.params import init_params


class Unit(nn.Module):
    def __init__(self):
        super().__init__()

    def init_params(self):
        self.apply(init_params())

    def save(self):
        return self.state_dict()

    def load(self, data):
        self.load_state_dict(data)

    # sometimes used by units that unfreeze after n epochs
    def end_of_epoch(self, epoch):
        pass

    def print_info(self):
        n_params = 0
        for p in self.parameters():
            n_params += p.numel()
        print('[INFO] built {} (total params: {})'.format(
            type(self).__name__,
            n_params,
        ))
        print(self)


class StaticUnit(nn.Module):
    def __init__(self):
        super().__init__()

    def init_params(self):
        pass

    def print_info(self):
        pass
