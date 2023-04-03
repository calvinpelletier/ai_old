#!/usr/bin/env python
from random import choice, random


def hidden0():
    return choice([
        '[128, 32, 8]',
        '[256, 64, 16]',
        '[128, 64, 32, 16, 8]',
    ])

def normbasic():
    return _str(choice(['batch', 'none']))

def relus():
    return _str(choice(['relu', 'lrelu', 'prelu']))

def bool():
    return repr(random() > 0.5)

def lr():
    return str(choice([0.01, 0.005, 0.001, 0.0005, 0.0001]))


def _str(x):
    return "'{}'".format(x)
