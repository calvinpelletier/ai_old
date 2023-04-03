#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from external.sg2 import training_stats
import numpy as np
from external.op import conv2d_gradfix


class LatentGanLoss:
    def __init__(self, cfg, trainer, device):
        self.trainer = trainer
        self.device = device

        self.r1_gamma = cfg.loss.D.gp.weight
        if self.r1_gamma == 'auto':
            self.r1_gamma = 0.0002 * (cfg.dataset.imsize ** 2) / \
                cfg.dataset.batch_size

    def accumulate_gradients(self, phase, real, seed, sync, gain):
        assert phase in ['G', 'Dmain', 'Dreg', 'Dboth']
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # G
        if phase == 'G':
            with torch.autograd.profiler.record_function('G_forward'):
                # gen
                latent = self.trainer.run_G(seed, sync=sync)

                # loss
                logits = self.trainer.run_D_for_latent(latent, sync=False)
                training_stats.report('loss/scores/fake', logits)
                training_stats.report('loss/signs/fake', logits.sign())
                gan_loss = torch.nn.functional.softplus(-logits)
                training_stats.report('loss/G/gan', gan_loss)

            with torch.autograd.profiler.record_function('G_backward'):
                gan_loss.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                latent = self.trainer.run_G(seed.detach(), sync=sync)
                gen_logits = self.trainer.run_D_for_latent(
                    latent,
                    sync=False, # synced by loss_Dreal
                )
                training_stats.report('loss/scores/fake', gen_logits)
                training_stats.report('loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else \
                'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real.detach().requires_grad_(do_Dr1)
                real_logits = self.trainer.run_D_for_img(real_img_tmp, sync=sync)
                training_stats.report('loss/scores/real', real_logits)
                training_stats.report('loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report('loss/D/gan', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), \
                            conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(
                            outputs=[real_logits.sum()],
                            inputs=[real_img_tmp],
                            create_graph=True,
                            only_inputs=True,
                        )[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('loss/D/gp_raw', r1_penalty)
                    training_stats.report('loss/D/gp_scaled', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (
                    real_logits * 0 + loss_Dreal + loss_Dr1
                ).mean().mul(gain).backward()
