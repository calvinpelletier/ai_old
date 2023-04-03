#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from external.sg2 import training_stats
import numpy as np
from external.op import conv2d_gradfix


class SimpleAaeLoss:
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
        self.no_face_loss = cfg.loss.G.rec.perceptual_type == 'ahanu'
        if self.no_face_loss:
            self.perceptual_weight = 1.
        else:
            self.perceptual_weight = 0.8
            self.face_weight = 0.1

    def accumulate_gradients(self, phase, real, sync, gain):
        assert phase in ['G_aae', 'Dmain', 'Dreg', 'Dboth']
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # G_aae
        if phase == 'G_aae':
            with torch.autograd.profiler.record_function('G_adv_forward'):
                # recreate image
                rec = self.trainer.run_G(
                    real=real.detach(),
                    do_req_losses=False,
                    sync=sync,
                )

                # gan loss
                rec_logits = self.trainer.run_D(rec, sync=False)
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
                results = self.trainer.run_G(
                    real=real.detach(),
                    do_req_losses=True,
                    sync=sync,
                    return_ws=do_delta_reg,
                )
                if do_delta_reg:
                    rec, ws, face_loss, perceptual_loss = results
                else:
                    rec, face_loss, perceptual_loss = results

                # rec loss
                pixel_loss = F.mse_loss(rec, real.detach())
                training_stats.report('loss/G/pixel', pixel_loss)
                training_stats.report('loss/G/perceptual', perceptual_loss)
                rec_loss = pixel_loss + \
                    perceptual_loss * self.perceptual_weight
                if not self.no_face_loss:
                    training_stats.report('loss/G/face', face_loss)
                    rec_loss += face_loss * self.face_weight

                # delta loss
                if do_delta_reg:
                    total_delta_loss = 0
                    w0 = ws[:, 0, :]
                    for i in range(1, ws.shape[1]):
                        delta = ws[:, i, :] - w0
                        delta_loss = torch.norm(delta, 2, dim=1).mean()
                        total_delta_loss += delta_loss
                    training_stats.report('loss/G/delta', total_delta_loss)
                    rec_loss += total_delta_loss * self.delta_reg_weight

                rec_loss *= self.overall_rec_weight
                training_stats.report('loss/G/rec', rec_loss)

            with torch.autograd.profiler.record_function('G_rec_backward'):
                rec_loss.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                rec = self.trainer.run_G(
                    real=real.detach(),
                    do_req_losses=False,
                    sync=sync,
                )
                gen_logits = self.trainer.run_D(
                    rec,
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
                real_logits = self.trainer.run_D(real_img_tmp, sync=sync)
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


class SimpleNonFaceAaeLoss:
    def __init__(self, cfg, trainer, device):
        self.trainer = trainer
        self.device = device

        self.r1_gamma = cfg.loss.D.gp.weight
        if self.r1_gamma == 'auto':
            self.r1_gamma = 0.0002 * (cfg.dataset.imsize ** 2) / \
                cfg.dataset.batch_size

        self.overall_rec_weight = cfg.loss.G.rec.weight

        # TODO: make configurable
        self.perceptual_weight = 1.

    def accumulate_gradients(self, phase, real, sync, gain):
        assert phase in ['G_aae', 'Dmain', 'Dreg', 'Dboth']
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # G_aae
        if phase == 'G_aae':
            with torch.autograd.profiler.record_function('G_adv_forward'):
                # recreate image
                rec = self.trainer.run_G(
                    real=real.detach(),
                    do_req_losses=False,
                    sync=sync,
                )

                # gan loss
                rec_logits = self.trainer.run_D(rec, sync=False)
                training_stats.report('loss/scores/fake', rec_logits)
                training_stats.report('loss/signs/fake', rec_logits.sign())
                gan_loss = torch.nn.functional.softplus(-rec_logits)
                training_stats.report('loss/G/gan', gan_loss)

            with torch.autograd.profiler.record_function('G_adv_backward'):
                gan_loss.mean().mul(gain).backward()

            with torch.autograd.profiler.record_function('G_rec_forward'):
                # recreate image
                rec, perceptual_loss = self.trainer.run_G(
                    real=real.detach(),
                    do_req_losses=True,
                    sync=sync,
                )

                # rec loss
                pixel_loss = F.mse_loss(rec, real.detach())
                training_stats.report('loss/G/pixel', pixel_loss)
                training_stats.report('loss/G/perceptual', perceptual_loss)
                rec_loss = pixel_loss + \
                    perceptual_loss * self.perceptual_weight
                rec_loss *= self.overall_rec_weight
                training_stats.report('loss/G/rec', rec_loss)

            with torch.autograd.profiler.record_function('G_rec_backward'):
                rec_loss.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                rec = self.trainer.run_G(
                    real=real.detach(),
                    do_req_losses=False,
                    sync=sync,
                )
                gen_logits = self.trainer.run_D(
                    rec,
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
                real_logits = self.trainer.run_D(real_img_tmp, sync=sync)
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


class SmartOutpaintLoss:
    def __init__(self, cfg, trainer, device):
        self.trainer = trainer
        self.device = device

        self.r1_gamma = cfg.loss.D.gp.weight
        if self.r1_gamma == 'auto':
            self.r1_gamma = 0.0002 * (cfg.dataset.imsize ** 2) / \
                cfg.dataset.batch_size

        self.rec_loss_enabled = cfg.loss.G.rec.enabled
        if self.rec_loss_enabled:
            self.rec_weight = cfg.loss.G.rec.weight

    def accumulate_gradients(self, phase, real, inpaint_mask, sync, gain):
        assert phase in ['G_aae', 'Dmain', 'Dreg', 'Dboth']
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # G_aae
        if phase == 'G_aae':
            with torch.autograd.profiler.record_function('G_adv_forward'):
                out = self.trainer.run_G(
                    real=real.detach(),
                    inpaint_mask=inpaint_mask.detach(),
                    do_req_losses=False,
                    sync=sync,
                )

                # gan loss
                logits = self.trainer.run_D(
                    out,
                    inpaint_mask.detach(),
                    sync=False,
                )
                training_stats.report('loss/scores/fake', logits)
                training_stats.report('loss/signs/fake', logits.sign())
                gan_loss = torch.nn.functional.softplus(-logits)
                training_stats.report('loss/G/gan', gan_loss)

            with torch.autograd.profiler.record_function('G_adv_backward'):
                gan_loss.mean().mul(gain).backward()

            if self.rec_loss_enabled:
                with torch.autograd.profiler.record_function('G_rec_forward'):
                    # recreate image
                    rec, perceptual_loss = self.trainer.run_G(
                        real=real.detach(),
                        inpaint_mask=inpaint_mask.detach(),
                        do_req_losses=True,
                        sync=sync,
                    )

                    # rec loss
                    pixel_loss = F.mse_loss(rec, real.detach())
                    training_stats.report('loss/G/pixel', pixel_loss)
                    training_stats.report('loss/G/perceptual', perceptual_loss)
                    rec_loss = pixel_loss + perceptual_loss
                    rec_loss *= self.rec_weight
                    training_stats.report('loss/G/rec', rec_loss)

                with torch.autograd.profiler.record_function('G_rec_backward'):
                    rec_loss.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                rec = self.trainer.run_G(
                    real=real.detach(),
                    inpaint_mask=inpaint_mask.detach(),
                    do_req_losses=False,
                    sync=sync,
                )
                gen_logits = self.trainer.run_D(
                    rec,
                    inpaint_mask.detach(),
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
                real_logits = self.trainer.run_D(
                    real_img_tmp,
                    inpaint_mask.detach(),
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
