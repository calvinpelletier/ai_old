#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from ai_old.nn.models.lerp.enc import dynamic_to_learned_const
from external.optimizer.ranger import Ranger
import external.sg2.misc as misc
from tqdm import tqdm


def finetune_enc_lerp(
    enc_lerper,
    base_enc,
    guide_enc,
    target_enc,
    mags,
    opt_type='adam',
    lr=0.05,
    n_iter=1000,
    return_losses=False,
    use_tqdm=False,
):
    bs1 = base_enc.shape[0]
    assert bs1 == 1
    bs2 = target_enc.shape[0]
    misc.assert_shape(base_enc, [bs1, 512, 4, 4])
    misc.assert_shape(guide_enc, [bs1, 512, 4, 4])
    misc.assert_shape(target_enc, [bs2, 512, 4, 4])
    misc.assert_shape(mags, [bs2])

    # enc lerp
    enc_lerp_lc, enc_lerp_g = dynamic_to_learned_const(
        enc_lerper,
        base_enc,
        guide_enc,
    )
    enc_lerp_lc = enc_lerp_lc.train().requires_grad_(True).to('cuda')
    enc_lerp_g = enc_lerp_g.eval().requires_grad_(False).to('cuda')

    # optimizer
    if opt_type == 'adam':
        opt = torch.optim.Adam(enc_lerp_lc.parameters(), lr=lr)
    elif opt_type == 'ranger':
        opt = Ranger(enc_lerp_lc.parameters(), lr=lr)
    else:
        raise Exception(opt_type)

    # finetune
    if return_losses:
        losses = []
    iterator = tqdm(range(n_iter)) if use_tqdm else range(n_iter)
    for i in iterator:
        opt.zero_grad()
        identity, base_w, delta = enc_lerp_lc()
        pred_enc = enc_lerp_g(
            base_enc.detach(),
            identity,
            base_w,
            delta,
            mags.detach(),
        )
        loss = F.mse_loss(pred_enc, target_enc.detach())
        if return_losses:
            losses.append(loss.item())
        loss.backward()
        opt.step()
    print(f'final enc lerp loss: {loss.item():.4f}')

    if return_losses:
        return enc_lerp_lc, enc_lerp_g, losses
    else:
        return enc_lerp_lc, enc_lerp_g
