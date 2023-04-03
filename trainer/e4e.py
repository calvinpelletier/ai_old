#!/usr/bin/env python3
import torch
from ai_old.trainer.rec import RecTrainer
import external.sg2.misc as misc


class E4eTrainer(RecTrainer):
    def __init__(self, cfg, rank, device):
        super().__init__(cfg, rank, device)
        self.cur_batch_num = 0


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

        self.cur_batch_num += 1
        if self.cur_batch_num % self.cfg.trainer.inc == 0 and \
                self.cur_batch_num >= self.cfg.trainer.cliff:
            self.G.e.increment_stage()


    def run_G(self, input, sync):
        G_g = self.ddp_modules['G_g']
        G_e = self.ddp_modules['G_e']
        face_loss_model = self.ddp_modules['arcface']
        perceptual_loss_model = self.ddp_modules['perceptual']

        # encode
        with misc.ddp_sync(G_e, sync):
            ws, delta_loss = G_e(input, return_delta_loss=True)

        # synthesize
        with misc.ddp_sync(G_g, sync):
            output = G_g(ws)

        # face loss
        with misc.ddp_sync(face_loss_model, sync):
            face_loss = face_loss_model(output, input)

        # perceptual loss
        with misc.ddp_sync(perceptual_loss_model, sync):
            perceptual_loss = perceptual_loss_model(output, input)

        return output, delta_loss, face_loss, perceptual_loss
