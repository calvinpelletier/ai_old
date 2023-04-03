#!/usr/bin/env python3
from ai_old.util.etc import print_
from PIL import Image
from ai_old.util.factory import build_dataset
import copy
import numpy as np
import torch
import torch.nn.functional as F
from ai_old.loss.perceptual.face import FaceIdLoss
import lpips
from ai_old.task.rec import RecTask
from ai_old.util.etc import create_img_row, normalized_tensor_to_pil_img


class Sg2DistillTask(RecTask):
    def sample(self, trainer, snapshot_data, cur_step):
        if self.rank != 0:
            return

        print('[INFO] saving samples...')

        batch_size = 32

        # build the test dataset
        dataset_core = build_dataset(self.cfg.dataset)
        test_set = dataset_core.get_test_set(
            batch_size,
            0, # seed
            self.rank,
            self.cfg.num_gpus,
            verbose=False,
        )

        # setup model
        model = copy.deepcopy(
            snapshot_data[trainer.get_model_key_for_eval()],
        ).eval().requires_grad_(False).to(self.device)

        # sampler
        sample_fn = trainer.get_sample_fn()

        # loop
        for batch in test_set:
            imgs1, imgs2 = sample_fn(
                model,
                batch,
                batch_size,
                self.device,
            )
            for id, img1, img2 in zip(batch['item_id'], imgs1, imgs2):
                create_img_row(
                    [
                        normalized_tensor_to_pil_img(img1),
                        normalized_tensor_to_pil_img(img2),
                    ],
                    self.cfg.dataset.imsize,
                ).save(self.samples_path.format(cur_step, id))


    def eval(self, trainer, snapshot_data, cur_step):
        print_(self.rank, '[INFO] calculating metrics...')

        if self.rank == 0:
            # build the val dataset
            batch_size = 32
            dataset_core = build_dataset(self.cfg.dataset)
            val_set = dataset_core.get_val_set(
                batch_size,
                0, # seed
                self.rank,
                self.cfg.num_gpus,
                verbose=False,
            )

            # setup models
            model = copy.deepcopy(
                snapshot_data[trainer.get_model_key_for_eval()],
            ).eval().requires_grad_(False).to(self.device)
            face_loss_model = trainer.face_loss_model
            if face_loss_model is None:
                face_loss_model = FaceIdLoss(
                    self.cfg.dataset.imsize).eval().requires_grad_(False).to(
                        self.device)
            if self.cfg.loss.G.rec.perceptual_type == 'lpips_alex':
                lpips_loss_model = trainer.perceptual_loss_model
            else:
                lpips_loss_model = lpips.LPIPS(
                    net='alex').eval().requires_grad_(False).to(self.device)

            # evaluator
            eval_fn = trainer.get_eval_fn()

            # calc metrics
            l2_pixel = 0.
            face_loss = 0.
            lpips_loss = 0.
            for batch in val_set:
                target, rec = eval_fn(model, batch, batch_size, self.device)
                l2_pixel += F.mse_loss(rec, target).mean()
                face_loss += face_loss_model(rec, target).mean()
                lpips_loss += lpips_loss_model(rec, target).mean()
            l2_pixel /= len(val_set)
            face_loss /= len(val_set)
            lpips_loss /= len(val_set)

            self.eval_metrics_writer.write_metric(
                cur_step,
                {
                    'l2_pixel': l2_pixel.item(),
                    'face': face_loss.item(),
                    'lpips': lpips_loss.item(),
                },
            )

            self.val_metrics_writer.write_metric(
                cur_step,
                {'todo': 0.},
            )


    def _get_eval_metric_names(self):
        return ['l2_pixel', 'face', 'lpips']


    def _get_val_metric_names(self):
        return ['todo']
