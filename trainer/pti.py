#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from tqdm import tqdm
from argparse import Namespace
from torchvision import transforms
from lpips import LPIPS

from external.pti.utils.models_utils import toogle_grad
from external.pti.criteria.localitly_regulizer import Space_Regulizer
from external.pti.projectors import w_projector
from external.e4e.models.psp import pSp
from external.pti.configs import paths_config, hyperparameters, global_config
from ai_old.util.pretrained import build_pretrained_sg2


class PtiTrainer:
    def __init__(self, device):
        self.device = device
        self.use_wandb = False
        global_config.pivotal_training_steps = 1
        global_config.training_step = 1

        # Initialize loss
        self.lpips_loss = LPIPS(
            net=hyperparameters.lpips_type).to(self.device).eval()

        self.restart_training()

    def restart_training(self):
        # Initialize networks
        self.G = build_pretrained_sg2(eval=False, device=self.device).train()
        toogle_grad(self.G, True)

        self.original_G = build_pretrained_sg2(eval=True, device=self.device)

        self.space_regulizer = Space_Regulizer(self.original_G, self.lpips_loss)
        self.optimizer = self.configure_optimizers()

    # def get_inversion(self, image):
    #     w_pivot = self.calc_inversions(image)
    #     w_pivot = w_pivot.to(self.device)
    #     return w_pivot

    # def calc_inversions(self, image):
    #     if hyperparameters.first_inv_type == 'w+':
    #         w = self.get_e4e_inversion(image)
    #
    #     else:
    #         id_image = torch.squeeze((image.to(self.device) + 1) / 2) * 255
    #         w = w_projector.project(
    #             self.G,
    #             id_image,
    #             device=torch.device(self.device),
    #             w_avg_samples=600,
    #             num_steps=hyperparameters.first_inv_steps,
    #             w_name='tmp',
    #             use_wandb=self.use_wandb,
    #         )
    #     return w

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.G.parameters(),
            lr=hyperparameters.pti_learning_rate,
        )
        return optimizer

    def calc_loss(self,
        generated_images,
        real_images,
        new_G,
        use_ball_holder,
        w_batch,
    ):
        loss = 0.0

        if hyperparameters.pt_l2_lambda > 0:
            l2_loss_val = F.mse_loss(generated_images, real_images)
            loss += l2_loss_val * hyperparameters.pt_l2_lambda

        if hyperparameters.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images, real_images)
            loss_lpips = torch.squeeze(loss_lpips)
            loss += loss_lpips * hyperparameters.pt_lpips_lambda

        if use_ball_holder and hyperparameters.use_locality_regularization:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(
                new_G,
                w_batch,
                use_wandb=self.use_wandb,
            )
            loss += ball_holder_loss_val

        return loss, l2_loss_val, loss_lpips

    def forward(self, w):
        generated_images = self.G.synthesis(w, noise_mode='const')
        return generated_images

    def train(self, image, w_pivot):
        use_ball_holder = True
        torch.cuda.empty_cache()
        self.restart_training()

        # w_pivot = self.calc_inversions(image)
        # w_pivot = w_pivot.to(self.device)

        real_images_batch = image.to(self.device)

        for i in range(hyperparameters.max_pti_steps):
            generated_images = self.forward(w_pivot)

            loss, l2_loss_val, loss_lpips = self.calc_loss(
                generated_images,
                real_images_batch,
                self.G,
                use_ball_holder,
                w_pivot,
            )

            self.optimizer.zero_grad()

            if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                break

            loss.backward()
            self.optimizer.step()

            use_ball_holder = global_config.training_step % \
                hyperparameters.locality_regularization_interval == 0

            global_config.training_step += 1

        return self.G
