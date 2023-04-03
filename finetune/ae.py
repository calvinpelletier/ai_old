#!/usr/bin/env python3
import torch
from ai_old.nn.models.ae.noise import dynamic_to_learned_const
from external.optimizer.ranger import Ranger
from ai_old.loss.imsim import SoloFaceImSimLoss
import external.sg2.misc as misc


def finetune_ae(
    ae,
    target,
    opt_type='adam',
    lr=0.4,
    n_iter=100,
    return_losses=False,
):
    misc.assert_shape(target, [None, 3, ae.imsize, ae.imsize])

    # ae
    ae_lc = dynamic_to_learned_const(
        ae,
        target,
    ).train().requires_grad_(True).to('cuda')
    ae = ae.eval().requires_grad_(False).to('cuda')

    # optimizer
    if opt_type == 'adam':
        opt = torch.optim.Adam(ae_lc.parameters(), lr=lr)
    elif opt_type == 'ranger':
        opt = Ranger(ae_lc.parameters(), lr=lr)
    else:
        raise Exception(opt_type)

    # loss function
    loss_fn = SoloFaceImSimLoss(
        ae.imsize,
        target.detach(),
    ).eval().requires_grad_(False).to('cuda')

    # finetune
    if return_losses:
        losses = []
    for i in range(n_iter):
        opt.zero_grad()
        enc = ae_lc()
        gen = ae.g(enc, noise_mode='const')
        loss = loss_fn(gen)
        if return_losses:
            losses.append(loss.item())
        loss.backward()
        opt.step()
    print(f'final ae loss: {loss.item():.4f}')

    if return_losses:
        return ae_lc, losses
    else:
        return ae_lc
