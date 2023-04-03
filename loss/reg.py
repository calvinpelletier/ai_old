#!/usr/bin/env python3
import torch


def gradient_penalty(d_real, y):
    if not d_real.requires_grad:
        return torch.tensor(0., device=d_real.device)

    outputs = [d_real]
    gradients = torch.autograd.grad(
        outputs=outputs,
        inputs=y,
        grad_outputs=list(map(lambda t: torch.ones(
            t.size(), device=y.device), outputs)),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    bs = y.shape[0]
    gradients = gradients.reshape(bs, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp
