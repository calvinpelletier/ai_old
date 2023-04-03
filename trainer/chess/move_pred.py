#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from ai_old.util.etc import print_
from ai_old.trainer.base import BaseTrainer
from external.sg2.misc import print_module_summary
from ai_old.util.factory import build_model
import external.sg2.misc as misc
from ai_old.loss.perceptual.face import FaceIdLoss
from ai_old.loss.clip import GenderSwapClipLoss, GenderSwapClipDirLoss
from ai_old.nn.models.seg.seg import colorize


class MovePredTrainer(BaseTrainer):
    def run_batch(self, batch, batch_idx, cur_step):
        # prep_data
        with torch.autograd.profiler.record_function('data_prep'):
            ctx = compact_board_enc_to_neural_board_enc(
                batch['board'], batch['move'])
            move = compact_move_enc_to_neural_move_enc(batch['move'])
            phase_ctx = ctx.split(self.batch_gpu)
            phase_move = move.split(self.batch_gpu)

        # run training phases
        for phase in self.phases:
            if batch_idx % phase.interval != 0:
                continue

            phase.init_gradient_accumulation()

            # accumulate gradients over multiple rounds
            for round_idx, (ctx, move) in enumerate(zip(phase_ctx, phase_move)):
                sync = (round_idx == self.batch_size // \
                    (self.batch_gpu * self.num_gpus) - 1)
                gain = phase.interval
                self.loss.accumulate_gradients(
                    phase=phase.name,
                    ctx=ctx,
                    move=move,
                    sync=sync,
                    gain=gain,
                )

            phase.update_params()


    def run_model(self, ctx, sync):
        model = self.ddp_modules['model']

        with misc.ddp_sync(model, sync):
            pred = model(ctx)

        return pred


    def get_eval_fn(self):
        def _eval_fn(model, batch, batch_size, device):
            ctx = compact_board_enc_to_neural_board_enc(
                batch['board'], batch['move'])
            gt = compact_move_enc_to_neural_move_enc(batch['move'])
            pred = model(ctx)
            return gt, pred
        return _eval_fn


    def get_sample_fn(self):
        # def _sample_fn(model, batch, batch_size, device):
        #     ctx = compact_board_enc_to_neural_board_enc(
        #         batch['board'], batch['move'])
        #     pred = model(ctx)
        #     return pred
        # return _sample_fn
        return None # TODO


    def _init_modules(self):
        cfg = self.cfg

        # build main model
        print_(self.rank, '[INFO] initializing model...')
        self.model = build_model(
            cfg,
            cfg.model,
        ).train().requires_grad_(False).to(self.device)

        # resume training
        assert not cfg.resume


    def _init_phases(self):
        print_(self.rank, '[INFO] initializing training phases...')
        self.phases = [self._build_loss_phase('main', self.model)]


    def get_modules_for_save(self):
        return [('model', self.model)]


    def _get_modules_for_distribution(self):
        return [
            ('model', self.model, True),
        ]


    def print_models(self):
        ctx = torch.empty([self.batch_gpu, 8, 8, 151], device=self.device)
        _ = print_module_summary(self.model, [ctx])
