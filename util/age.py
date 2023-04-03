#!/usr/bin/env python3
import torch
from random import randint
import math

MAX_AGE = 80


def rand_age_from_range(lower, upper):
    return randint(max(1, lower), min(MAX_AGE, upper))


def encode_age(age):
    enc = [1] * age + [0] * (MAX_AGE - age)
    return torch.tensor(enc)


def scale_age(age):
    return 10 * math.log(age + 6) - 18


def scale_and_encode_age(age):
    scaled = int(scale_age(age))
    max_age = get_max_age(is_scaled=True)
    enc = [1] * scaled + [0] * (max_age - scaled)
    return torch.tensor(enc)


def get_max_age(is_scaled=False):
    if is_scaled:
        return int(scale_age(MAX_AGE))
    return MAX_AGE


def unscale_age(scaled_age):
    return int(math.exp((scaled_age + 18) / 10) - 6)
