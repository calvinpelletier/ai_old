#!/usr/bin/env python3
import torch
import torch.nn as nn
from time import time
from ai_old.loss.base import BaseLoss
from random import random


class ComboLoss(BaseLoss):
    def __init__(self, loss_conf, is_for_eval=False):
        super().__init__()
        self.loss_conf = loss_conf
        self.is_for_eval = is_for_eval
        self.sublosses = nn.ModuleList()

        # for sublosses that dont happen every step
        self.last_values = {}
        self.last_times = {}
        self.requires_grad_for_y_ = False

    def create_subloss(self, name, fn, ents, requires_grad_for_y=False):
        if self.is_for_eval:
            self.sublosses.append(ComboLoss.Subloss(name, fn, ents, 1, None))
            return

        if requires_grad_for_y:
            self.requires_grad_for_y_ = True

        if hasattr(self.loss_conf, name):
            conf = getattr(self.loss_conf, name)

            if isinstance(conf.weight, list):
                weight = _WarmupValue(conf.weight)
            else:
                if conf.weight < 0.:
                    raise ValueError('no support for negative weight losses')
                elif conf.weight == 0.:
                    print(f'[WARNING] loss {name} has weight 0, skipping')
                    return
                weight = _StaticValue(conf.weight)

            # see self._should_calc_subloss for freq/prob explanation
            freq = conf.freq if hasattr(conf, 'freq') else None
            prob = conf.prob if hasattr(conf, 'prob') else None
            assert prob is None or freq is None # cant have both freq and prob

            if freq is not None:
                # calc every N steps offset by M steps
                # format: N (default M=0) or [N, M]
                if not isinstance(freq, list):
                    freq = [freq, 0]

            if prob is not None:
                if isinstance(prob, list):
                    prob = _WarmupValue(prob)
                else:
                    prob = _StaticValue(prob)

            if prob is not None or freq is not None:
                # init with dummy values (otherwise problems with --cont)
                self.last_values[name] = torch.tensor(0.)
                self.last_times[name] = torch.tensor(0.)

            self.sublosses.append(ComboLoss.Subloss(
                name, fn, ents, weight, freq, prob))
        else:
            print(f'[WARNING] loss {name} not in config, skipping')

    def has_multiple_losses(self):
        return True

    def get_loss_names(self):
        return [sl.name for sl in self.sublosses]

    def requires_grad_for_y(self):
        return self.requires_grad_for_y_

    def forward(self, ents, batch_num=None):
        return self.calc_and_sum_sublosses(ents, batch_num=batch_num)

    def calc_and_sum_sublosses(self, ents, batch_num=None):
        total_loss = 0.0
        subloss_values = {}
        subloss_times = {}

        for sl in self.sublosses:
            if self._should_calc_subloss(sl, batch_num):
                # calc and time subloss
                start = time()
                val = sl.fn(*[ents[ent] for ent in sl.ents])
                subloss_times[sl.name] = time() - start
                total_loss += val * sl.weight.get(batch_num)
                subloss_values[sl.name] = val.clone().detach()

                if sl.freq is not None or sl.prob is not None:
                    if sl.freq is not None:
                        subloss_times[sl.name] /= sl.freq[0]
                    else:
                        # sl.prob is not None
                        subloss_times[sl.name] *= sl.prob.get(batch_num)

                    # save info for later (because verbose might not align)
                    self.last_values[sl.name] = subloss_values[sl.name]
                    self.last_times[sl.name] = subloss_times[sl.name]
            else:
                # pull from last (only used for printing)
                subloss_values[sl.name] = self.last_values[sl.name]
                subloss_times[sl.name] = self.last_times[sl.name]

        return total_loss, subloss_values, subloss_times

    def _should_calc_subloss(self, sl, batch_num):
        # freq subloss (calc once per N steps)
        if sl.freq is not None:
            return batch_num % sl.freq[0] == sl.freq[1]

        # prob subloss (calc randomly with probability X)
        if sl.prob is not None:
            return random() < sl.prob.get(batch_num)

        # normal subloss (calc every step)
        return True

    # basically a namedtuple but needs to extend nn.Module because fn is a torch
    # module
    class Subloss(nn.Module):
        def __init__(self, name, fn, ents, weight, freq, prob):
            super().__init__()
            self.name = name
            self.fn = fn
            self.ents = ents
            self.weight = weight
            self.freq = freq
            self.prob = prob


class BinaryLoss(nn.Module):
    def __init__(self, loss_conf):
        super().__init__()
        self.pred_key = loss_conf.pred_key
        self.target_key = loss_conf.target_key
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, ents):
        loss = self.loss(ents[self.pred_key], ents[self.target_key])
        return loss, None, None # loss, sublosses, subloss_times


class SumLoss(nn.Module):
    def forward(self, x):
        return torch.sum(x)


class _StaticValue:
    def __init__(self, val):
        self.val = val

    def get(self, _batch_num):
        return self.val


class _WarmupValue:
    def __init__(self, conf):
        self.low, self.high, self.dur = conf
        self.range = self.high - self.low

    def get(self, batch_num):
        return self.low + self.range * (batch_num / self.dur)
