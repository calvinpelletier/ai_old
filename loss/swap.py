#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from external.sg2 import training_stats
from external.op import conv2d_gradfix


class SynthOnlySwapLoss:
    def __init__(self, cfg, trainer, device):
        self.trainer = trainer
        self.device = device

    def accumulate_gradients(self,
        phase,
        x_img,
        x_gender,
        y_img,
        y_gender,
        sync,
        gain,
    ):
        assert phase == 'swap'

        with torch.autograd.profiler.record_function('swap_forward'):
            x_latent, y_latent, x_latent_pred, y_latent_pred = \
                self.trainer.run_G(
                    x_img,
                    x_gender,
                    y_img,
                    y_gender,
                    sync,
                )
            loss = F.mse_loss(x_latent_pred, x_latent).mean() + \
                F.mse_loss(y_latent_pred, y_latent).mean()
            training_stats.report('loss/swap', loss)

        with torch.autograd.profiler.record_function('swap_backward'):
            loss.mul(gain).backward()


class AdversarialSwapLoss:
    def __init__(self, cfg, trainer, device):
        self.trainer = trainer
        self.device = device

        self.r1_gamma = cfg.loss.D.gp.weight
        if self.r1_gamma == 'auto':
            self.r1_gamma = 0.0002 * (cfg.dataset.imsize ** 2) / \
                cfg.dataset.batch_size

        self.adv_weight = cfg.loss.G.adv.weight

    def accumulate_gradients(self,
        phase,
        real_img,
        real_gender,
        x_img,
        x_gender,
        y_img,
        y_gender,
        sync,
        gain,
    ):
        assert phase in ['G', 'Dmain', 'Dreg', 'Dboth']
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        if phase == 'G':
            with torch.autograd.profiler.record_function('G_real_forward'):
                real_swap_latent = self.trainer.run_G_real(
                    real_img,
                    real_gender.detach(),
                    sync,
                )
                real_swap_gender = -1. * real_gender + 1.
                logits = self.trainer.run_D_for_latent(
                    real_swap_latent,
                    real_swap_gender,
                    sync=False,
                )
                training_stats.report('loss/scores/fake', logits)
                training_stats.report('loss/signs/fake', logits.sign())
                adv_loss = torch.nn.functional.softplus(-logits)
                training_stats.report('loss/G/adv', adv_loss)

            with torch.autograd.profiler.record_function('G_real_backward'):
                adv_loss.mean().mul(gain).backward()

            with torch.autograd.profiler.record_function('G_synth_forward'):
                x_latent, y_latent, x_latent_pred, y_latent_pred = \
                    self.trainer.run_G_synth(
                        x_img,
                        x_gender,
                        y_img,
                        y_gender,
                        sync,
                    )
                loss = F.mse_loss(x_latent_pred, x_latent).mean() + \
                    F.mse_loss(y_latent_pred, y_latent).mean()
                training_stats.report('loss/swap', loss)

            with torch.autograd.profiler.record_function('G_synth_backward'):
                loss.mul(gain).backward()

        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                real_swap_latent = self.trainer.run_G_real(
                    real_img.detach(),
                    real_gender.detach(),
                    sync,
                )
                real_swap_gender = -1. * real_gender + 1.
                gen_logits = self.trainer.run_D_for_latent(
                    real_swap_latent,
                    real_swap_gender,
                    sync=False, # synced by loss_Dreal
                )
                training_stats.report('loss/scores/fake', gen_logits)
                training_stats.report('loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else \
                'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_gender_tmp = real_gender.detach().requires_grad_(do_Dr1)
                real_logits = self.trainer.run_D_for_img(
                    real_img_tmp,
                    real_gender_tmp,
                    sync=sync,
                )
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
                            inputs=[real_img_tmp, real_gender_tmp],
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
