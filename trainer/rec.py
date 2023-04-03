#!/usr/bin/env python3
import torch
from ai_old.util.etc import print_
from ai_old.trainer.base import BaseTrainer
from external.sg2.misc import print_module_summary
from ai_old.util.factory import build_model, build_augpipe
import copy
from external.sg2 import training_stats
import external.sg2.misc as misc
import numpy as np
import external.sg2.legacy as legacy
import os
from ai_old.loss.perceptual.face import FaceIdLoss
import lpips
from ai_old.loss.perceptual.trad import PerceptualLoss
from ai_old.loss.perceptual.ahanu import AhanuPercepLoss


class RecTrainer(BaseTrainer):
    def run_batch(self, batch, batch_idx, cur_step):
        # prep_data
        with torch.autograd.profiler.record_function('data_prep'):
            phase_real_img = batch['y']
            phase_real_img = (
                phase_real_img.to(self.device).to(torch.float32) / 127.5 - 1
            ).split(self.batch_gpu)

        # run training phases
        for phase in self.phases:
            if batch_idx % phase.interval != 0:
                continue

            phase.init_gradient_accumulation()

            # accumulate gradients over multiple rounds
            for round_idx, real_img in enumerate(phase_real_img):
                sync = (round_idx == self.batch_size // \
                    (self.batch_gpu * self.num_gpus) - 1)
                gain = phase.interval
                self.loss.accumulate_gradients(
                    phase=phase.name,
                    real_img=real_img,
                    sync=sync,
                    gain=gain,
                )

            phase.update_params()


    def run_G(self, input, sync):
        G_g = self.ddp_modules['G_g']
        G_e = self.ddp_modules['G_e']

        if self.G.intermediate == 'enc_plus_z':
            # encode
            with misc.ddp_sync(G_e, sync):
                encoding, z = G_e(input)

            # synthesize
            with misc.ddp_sync(G_g, sync):
                output = G_g(encoding, z)
        elif self.G.intermediate in ['modspace', 'enc', 'zspace']:
            # encode
            with misc.ddp_sync(G_e, sync):
                ws = G_e(input)

            # synthesize
            with misc.ddp_sync(G_g, sync):
                output = G_g(ws)
        else:
            raise Exception(self.G.intermediate)

        # face loss
        if hasattr(self.cfg.loss, 'face'):
            face_loss_model = self.ddp_modules['arcface']
            with misc.ddp_sync(face_loss_model, sync):
                face_loss = face_loss_model(output, input.detach())
        else:
            face_loss = None

        # perceptual loss
        if hasattr(self.cfg.loss, 'perceptual'):
            perceptual_loss_model = self.ddp_modules['perceptual']
            with misc.ddp_sync(perceptual_loss_model, sync):
                perceptual_loss = perceptual_loss_model(output, input.detach())
        else:
            perceptual_loss = None

        return output, face_loss, perceptual_loss


    def get_eval_fn(self):
        def _eval_fn(model, batch, batch_size, device):
            input = batch['y'].to(device).to(torch.float32) / 127.5 - 1
            rec = model(input)
            return input, rec
        return _eval_fn


    def get_sample_fn(self):
        def _sample_fn(model, batch, batch_size, device):
            input = batch['y'].to(device).to(torch.float32) / 127.5 - 1
            rec = model(input)
            rec = (rec * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return rec
        return _sample_fn


    def get_model_key_for_eval(self):
        return 'G'


    def _init_modules(self):
        cfg = self.cfg

        # build main model
        print_(self.rank, '[INFO] initializing model...')
        self.G = build_model(
            cfg,
            cfg.model,
        ).train().requires_grad_(False).to(self.device)

        # face loss
        if hasattr(cfg.loss, 'face'):
            self.face_loss_model = FaceIdLoss(
                cfg.dataset.imsize).eval().requires_grad_(False).to(self.device)

        # perceptual loss
        if hasattr(cfg.loss, 'perceptual'):
            perceptual_type = cfg.loss.perceptual.type
            if perceptual_type == 'lpips':
                raise Exception('lpips_alex or lpips_vgg')
            elif perceptual_type == 'lpips_alex':
                percep = lpips.LPIPS(net='alex')
            elif perceptual_type == 'lpips_vgg':
                percep = lpips.LPIPS(net='vgg')
            elif perceptual_type == 'trad':
                percep = PerceptualLoss()
            elif perceptual_type == 'ahanu':
                percep = AhanuPercepLoss(cfg.loss.perceptual.version)
            else:
                raise Exception(perceptual_type)
            percep = percep.eval().requires_grad_(False).to(self.device)
            self.perceptual_loss_model = percep

        # resume training
        assert not cfg.resume


    def _init_phases(self):
        print_(self.rank, '[INFO] initializing training phases...')
        self.phases = [self._build_loss_phase('rec', self.G)]


    def get_modules_for_save(self):
        return [('G', self.G)]


    def _get_modules_for_distribution(self):
        modules = [
            ('G_g', self.G.g, False),
            ('G_e', self.G.e, True),
        ]
        if hasattr(self.cfg.loss, 'face'):
            modules.append(('arcface', self.face_loss_model, False))
        if hasattr(self.cfg.loss, 'perceptual'):
            modules.append(('perceptual', self.perceptual_loss_model, False))
        return modules


    def print_models(self):
        input = torch.empty(
            [self.batch_gpu, 3, self.G.imsize, self.G.imsize],
            device=self.device,
        )
        _ = print_module_summary(self.G, [input])


# class BlendRecTrainer(RecTrainer):
#     def run_batch(self, batch, batch_idx, cur_step):
#         # prep_data
#         with torch.autograd.profiler.record_function('data_prep'):
#             phase_full = batch['full']
#             phase_full = (
#                 phase_full.to(self.device).to(torch.float32) / 127.5 - 1
#             ).split(self.batch_gpu)
#
#             phase_fg = batch['fg']
#             phase_fg = (
#                 phase_fg.to(self.device).to(torch.float32) / 127.5 - 1
#             ).split(self.batch_gpu)
#
#             phase_ibg = batch['ibg']
#             phase_ibg = (
#                 phase_ibg.to(self.device).to(torch.float32) / 127.5 - 1
#             ).split(self.batch_gpu)
#
#         # run training phases
#         for phase in self.phases:
#             if batch_idx % phase.interval != 0:
#                 continue
#
#             phase.init_gradient_accumulation()
#
#             # accumulate gradients over multiple rounds
#             for round_idx, (full, fg, ibg) in enumerate(zip(
#                     phase_full, phase_fg, phase_ibg)):
#                 sync = (round_idx == self.batch_size // \
#                     (self.batch_gpu * self.num_gpus) - 1)
#                 gain = phase.interval
#                 self.loss.accumulate_gradients(
#                     phase=phase.name,
#                     full=full,
#                     fg=fg,
#                     ibg=ibg,
#                     sync=sync,
#                     gain=gain,
#                 )
#
#             phase.update_params()
#
#
#     def run_G(self, full, fg, ibg, sync):
#         G_g = self.ddp_modules['G_g']
#         G_e = self.ddp_modules['G_e']
#         face_loss_model = self.ddp_modules['arcface']
#         lpips_loss_model = self.ddp_modules['lpips']
#
#         # encode
#         with misc.ddp_sync(G_e, sync):
#             ws, ibg_encs = G_e(fg, ibg)
#
#         # synthesize
#         with misc.ddp_sync(G_g, sync):
#             output = G_g(ibg_encs, ws)
#
#         # face loss
#         with misc.ddp_sync(face_loss_model, sync):
#             face_loss = face_loss_model(output, full)
#
#         # lpips loss
#         with misc.ddp_sync(lpips_loss_model, sync):
#             lpips_loss = lpips_loss_model(output, full)
#
#         return output, face_loss, lpips_loss
#
#
#     def get_eval_fn(self):
#         def _eval_fn(model, batch, batch_size, device):
#             fg = batch['fg'].to(device).to(torch.float32) / 127.5 - 1
#             ibg = batch['ibg'].to(device).to(torch.float32) / 127.5 - 1
#             rec = model(fg, ibg)
#             rec = (rec * 127.5 + 128).clamp(0, 255).to(torch.uint8)
#             return rec
#         return _eval_fn
#
#
#     def get_sample_fn(self):
#         def _sample_fn(model, batch, batch_size, device):
#             fg = batch['fg'].to(device).to(torch.float32) / 127.5 - 1
#             ibg = batch['ibg'].to(device).to(torch.float32) / 127.5 - 1
#             rec = model(fg, ibg)
#             rec = (rec * 127.5 + 128).clamp(0, 255).to(torch.uint8)
#             return rec
#         return _sample_fn
#
#
#     def _get_modules_for_distribution(self):
#         return [
#             ('G_g', self.G.g, False),
#             ('G_e', self.G.e, True),
#             ('arcface', self.face_loss_model, False),
#             ('lpips', self.lpips_loss_model, False),
#         ]
#
#
#     def print_models(self):
#         fg = torch.empty(
#             [self.batch_gpu, 3, self.G.imsize, self.G.imsize],
#             device=self.device,
#         )
#         ibg = torch.empty(
#             [self.batch_gpu, 3, self.G.imsize, self.G.imsize],
#             device=self.device,
#         )
#         _ = print_module_summary(self.G, [fg, ibg])
#
#
# class OldBlendRecTrainer(RecTrainer):
#     def run_batch(self, batch, batch_idx, cur_step):
#         # prep_data
#         with torch.autograd.profiler.record_function('data_prep'):
#             phase_full = batch['full']
#             phase_full = (
#                 phase_full.to(self.device).to(torch.float32) / 127.5 - 1
#             ).split(self.batch_gpu)
#
#             phase_fg = batch['fg']
#             phase_fg = (
#                 phase_fg.to(self.device).to(torch.float32) / 127.5 - 1
#             ).split(self.batch_gpu)
#
#             phase_ibg = batch['ibg']
#             phase_ibg = (
#                 phase_ibg.to(self.device).to(torch.float32) / 127.5 - 1
#             ).split(self.batch_gpu)
#
#         # run training phases
#         for phase in self.phases:
#             if batch_idx % phase.interval != 0:
#                 continue
#
#             phase.init_gradient_accumulation()
#
#             # accumulate gradients over multiple rounds
#             for round_idx, (full, fg, ibg) in enumerate(zip(
#                     phase_full, phase_fg, phase_ibg)):
#                 sync = (round_idx == self.batch_size // \
#                     (self.batch_gpu * self.num_gpus) - 1)
#                 gain = phase.interval
#                 self.loss.accumulate_gradients(
#                     phase=phase.name,
#                     full=full,
#                     fg=fg,
#                     ibg=ibg,
#                     sync=sync,
#                     gain=gain,
#                 )
#
#             phase.update_params()
#
#
#     def run_G(self, full, fg, ibg, sync):
#         G_g = self.ddp_modules['G_g']
#         G_e = self.ddp_modules['G_e']
#         G_e_ibg = self.ddp_modules['G_e_ibg']
#         face_loss_model = self.ddp_modules['arcface']
#         lpips_loss_model = self.ddp_modules['lpips']
#
#         # encode fg
#         with misc.ddp_sync(G_e, sync):
#             w = G_e(fg)
#
#         # encode ibg
#         with misc.ddp_sync(G_e_ibg, sync):
#             ibg_encs = G_e_ibg(ibg, w)
#
#         # synthesize
#         with misc.ddp_sync(G_g, sync):
#             ws = w.unsqueeze(1).repeat([1, self.G.num_ws, 1])
#             output = G_g(ibg_encs, ws)
#
#         # face loss
#         with misc.ddp_sync(face_loss_model, sync):
#             face_loss = face_loss_model(output, full)
#
#         # lpips loss
#         with misc.ddp_sync(lpips_loss_model, sync):
#             lpips_loss = lpips_loss_model(output, full)
#
#         return output, face_loss, lpips_loss
#
#
#     def get_eval_fn(self):
#         def _eval_fn(model, batch, batch_size, device):
#             fg = batch['fg'].to(device).to(torch.float32) / 127.5 - 1
#             ibg = batch['ibg'].to(device).to(torch.float32) / 127.5 - 1
#             rec = model(fg, ibg)
#             rec = (rec * 127.5 + 128).clamp(0, 255).to(torch.uint8)
#             return rec
#         return _eval_fn
#
#
#     def get_sample_fn(self):
#         def _sample_fn(model, batch, batch_size, device):
#             fg = batch['fg'].to(device).to(torch.float32) / 127.5 - 1
#             ibg = batch['ibg'].to(device).to(torch.float32) / 127.5 - 1
#             rec = model(fg, ibg)
#             rec = (rec * 127.5 + 128).clamp(0, 255).to(torch.uint8)
#             return rec
#         return _sample_fn
#
#
#     def _get_modules_for_distribution(self):
#         return [
#             ('G_g', self.G.g, False),
#             ('G_e', self.G.e, True),
#             ('G_e_ibg', self.G.e_ibg, True),
#             ('arcface', self.face_loss_model, False),
#             ('lpips', self.lpips_loss_model, False),
#         ]
#
#
#     def print_models(self):
#         fg = torch.empty(
#             [self.batch_gpu, 3, self.G.imsize, self.G.imsize],
#             device=self.device,
#         )
#         ibg = torch.empty(
#             [self.batch_gpu, 3, self.G.imsize, self.G.imsize],
#             device=self.device,
#         )
#         _ = print_module_summary(self.G, [fg, ibg])
