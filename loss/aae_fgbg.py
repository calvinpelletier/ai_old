#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from external.sg2 import training_stats
import numpy as np
from external.op import conv2d_gradfix


class FgBgAaeLoss:
    def __init__(self, cfg, trainer, device):
        self.trainer = trainer
        self.device = device

        self.r1_gamma = cfg.loss.D.gp.weight
        if self.r1_gamma == 'auto':
            self.r1_gamma = 0.0002 * (cfg.dataset.imsize ** 2) / \
                cfg.dataset.batch_size

        self.overall_rec_weight = cfg.loss.G.rec.weight

        if hasattr(cfg.loss.G, 'delta_reg'):
            self.delta_reg_weight = cfg.loss.G.delta_reg.weight
        else:
            self.delta_reg_weight = None

        # TODO: make configurable
        self.perceptual_weight = 0.8
        self.face_weight = 0.1

    def accumulate_gradients(self, phase, real_img, real_seg, sync, gain):
        assert phase in ['G_aae', 'Dmain', 'Dreg', 'Dboth']
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # G_aae
        if phase == 'G_aae':
            with torch.autograd.profiler.record_function('G_adv_forward'):
                # recreate image
                rec, seg = self.trainer.run_G(
                    real_img=real_img.detach(),
                    do_rec_losses=False,
                    sync=sync,
                )

                # gan loss
                rec_logits = self.trainer.run_D(rec, seg, sync=False)
                training_stats.report('loss/scores/fake', rec_logits)
                training_stats.report('loss/signs/fake', rec_logits.sign())
                gan_loss = torch.nn.functional.softplus(-rec_logits)
                training_stats.report('loss/G/gan', gan_loss)

            with torch.autograd.profiler.record_function('G_adv_backward'):
                gan_loss.mean().mul(gain).backward()

            with torch.autograd.profiler.record_function('G_rec_forward'):
                do_delta_reg = self.delta_reg_weight is not None and \
                    self.delta_reg_weight > 0.

                # recreate image
                rec, _seg, face_loss, perceptual_loss = self.trainer.run_G(
                    real_img=real_img.detach(),
                    do_rec_losses=True,
                    sync=sync,
                )

                # rec loss
                pixel_loss = F.mse_loss(rec, real_img.detach())
                training_stats.report('loss/G/pixel', pixel_loss)
                training_stats.report('loss/G/face', face_loss)
                training_stats.report('loss/G/perceptual', perceptual_loss)
                rec_loss = pixel_loss + \
                    perceptual_loss * self.perceptual_weight + \
                    face_loss * self.face_weight
                rec_loss *= self.overall_rec_weight
                training_stats.report('loss/G/rec', rec_loss)

            with torch.autograd.profiler.record_function('G_rec_backward'):
                rec_loss.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                rec, seg = self.trainer.run_G(
                    real_img=real_img.detach(),
                    do_rec_losses=False,
                    sync=sync,
                )
                gen_logits = self.trainer.run_D(
                    rec,
                    seg,
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
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_seg_tmp = real_seg.detach().requires_grad_(do_Dr1)
                real_logits = self.trainer.run_D(
                    real_img_tmp,
                    real_seg_tmp,
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
                            inputs=[real_img_tmp, real_seg_tmp],
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
