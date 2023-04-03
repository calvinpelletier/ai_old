#!/usr/bin/env python3
from ai_old.trainer.rec import RecTrainer
import external.sg2.misc as misc


class RestylePspTrainer(RecTrainer):
    def __init__(self, cfg, rank, device):
        super().__init__(cfg, rank, device)

        self.G.calc_avg_img()

    def run_G(self, input_img, base_img, base_ws, sync):
        G_g = self.ddp_modules['G_g']
        G_e = self.ddp_modules['G_e']
        face_loss_model = self.ddp_modules['arcface']
        perceptual_loss_model = self.ddp_modules['perceptual']

        # encode
        with misc.ddp_sync(G_e, sync):
            ws = G_e(input_img, base_img, base_ws)

        # synthesize
        with misc.ddp_sync(G_g, sync):
            output_img = G_g(ws)

        # face loss
        with misc.ddp_sync(face_loss_model, sync):
            face_loss = face_loss_model(output_img, input_img)

        # perceptual loss
        with misc.ddp_sync(perceptual_loss_model, sync):
            perceptual_loss = perceptual_loss_model(output_img, input_img)

        return output_img, ws, face_loss, perceptual_loss
