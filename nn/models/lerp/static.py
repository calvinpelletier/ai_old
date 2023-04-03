#!/usr/bin/env python3
import torch
import torch.nn as nn


def convert_dynamic_lerper_to_static(dynamic, w, gender, type):
    new_w = dynamic(w, gender, magnitude=1.)
    delta = new_w - w
    assert delta.shape == (1, 18, 512)

    if hasattr(dynamic, 'lerper'):
        coarse_enabled = not dynamic.lerper.opts.no_coarse_mapper
        medium_enabled = not dynamic.lerper.opts.no_medium_mapper
        fine_enabled = not dynamic.lerper.opts.no_fine_mapper
    else:
        coarse_enabled = dynamic.coarse_enabled
        medium_enabled = dynamic.medium_enabled
        fine_enabled = dynamic.fine_enabled

    if type == 'full':
        cls = StaticLearnedWPlusLerp
    elif type == 'levels':
        cls = StaticLearnedWPlusLevelsLerp
    else:
        raise Exception(type)

    return cls(
        init=delta.squeeze(),
        coarse_enabled=coarse_enabled,
        medium_enabled=medium_enabled,
        fine_enabled=fine_enabled,
    )


class StaticLearnedWPlusLerp(nn.Module):
    def __init__(self,
        w_dims=512,
        n_latents=18,
        init=None,
        coarse_enabled=True,
        medium_enabled=True,
        fine_enabled=True,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.coarse_enabled = coarse_enabled
        self.medium_enabled = medium_enabled
        self.fine_enabled = fine_enabled

        for i in range(n_latents):
            if self.is_idx_enabled(i):
                if init is None:
                    val = torch.randn([w_dims])
                else:
                    val = init[i].clone().detach()
                    assert val.shape == (512,)
                setattr(self, f'latent{i}', nn.Parameter(val))

    def is_idx_enabled(self, idx):
        if idx < 4:
            return self.coarse_enabled
        elif idx < 8:
            return self.medium_enabled
        else:
            return self.fine_enabled

    def forward(self, w, _gender, magnitude=1.):
        delta = torch.zeros_like(w)
        for i in range(self.n_latents):
            if self.is_idx_enabled(i):
                delta[:, i, :] = getattr(self, f'latent{i}')
        return w + delta * magnitude


class StaticLearnedWPlusLevelsLerp(nn.Module):
    def __init__(self,
        w_dims=512,
        n_latents=18,
        init=None,
        coarse_enabled=True,
        medium_enabled=True,
        fine_enabled=True,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.coarse_enabled = coarse_enabled
        self.medium_enabled = medium_enabled
        self.fine_enabled = fine_enabled

        if self.coarse_enabled:
            self.coarse = self._build_level(init, 0, 4)
        if self.medium_enabled:
            self.medium = self._build_level(init, 4, 8)
        if self.fine_enabled:
            self.fine = self._build_level(init, 8, 18)

    def _build_level(self, init, i, j):
        if init is None:
            val = torch.randn([w_dims])
        else:
            val = init[i:j, :].mean(dim=0).detach()
            assert val.shape == (512,)
        return nn.Parameter(val)

    def forward(self, w, _gender, magnitude=1.):
        delta = torch.zeros_like(w)
        if self.coarse_enabled:
            delta[:, :4, :] = self.coarse.unsqueeze(dim=0).repeat(4, 1)
        if self.medium_enabled:
            delta[:, 4:8, :] = self.medium.unsqueeze(dim=0).repeat(4, 1)
        if self.fine_enabled:
            delta[:, 8:, :] = self.fine.unsqueeze(dim=0).repeat(10, 1)
        return w + delta * magnitude
