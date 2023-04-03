#!/usr/bin/env python3
import torch
from external.sg2 import training_stats
import numpy as np
from external.op import conv2d_gradfix


class FgBgStyleGanLoss:
    def __init__(self,
        cfg,
        trainer,
        device,
    ):
        self.trainer = trainer
        self.device = device

        self.r1_gamma = cfg.loss.D.gp.weight
        if self.r1_gamma == 'auto':
            self.r1_gamma = 0.0002 * (cfg.dataset.imsize ** 2) / \
                cfg.dataset.batch_size

        self.pl_batch_shrink = cfg.loss.G.ppl.batch_shrink
        self.pl_decay = cfg.loss.G.ppl.decay
        self.pl_weight = cfg.loss.G.ppl.weight
        self.pl_mean = torch.zeros([], device=device)

    def accumulate_gradients(self,
        phase,
        real_img,
        real_seg,
        z_fg,
        z_bg,
        sync,
        gain,
    ):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws, gen_seg = self.trainer.run_G(
                    z_fg,
                    z_bg,
                    sync=(sync and not do_Gpl),
                )
                gen_logits = self.trainer.run_D(gen_img, gen_seg, sync=False)
                training_stats.report('loss/scores/fake', gen_logits)
                training_stats.report('loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report('loss/G/gan', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = z_fg.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws, _seg = self.trainer.run_G(
                    z_fg[:batch_size],
                    z_bg[:batch_size],
                    sync=sync,
                )
                pl_noise = torch.randn_like(gen_img) / \
                    np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), \
                        conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(
                        outputs=[(gen_img * pl_noise).sum()],
                        inputs=[gen_ws],
                        create_graph=True,
                        only_inputs=True,
                    )[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('loss/G/ppl_raw', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('loss/G/ppl_scaled', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws, gen_seg = self.trainer.run_G(
                    z_fg,
                    z_bg,
                    sync=False,
                )
                gen_logits = self.trainer.run_D(
                    gen_img,
                    gen_seg,
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
