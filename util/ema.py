#!/usr/bin/env python3


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        return old * self.beta + (1 - self.beta) * new
