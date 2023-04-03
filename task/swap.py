#!/usr/bin/env python3
import ai_old.constants as c
from ai_old.util.etc import print_
from metrics import metric_main
import os
from ai_old.util.metrics_writer import MetricsWriter
from PIL import Image
from ai_old.util.factory import build_dataset
import copy
import numpy as np
import torch
import torch.nn.functional as F
from ai_old.loss.perceptual.face import FaceLoss
from ai_old.loss.perceptual.lpips import LpipsLoss
from ai_old.loss.gender import GenderLoss


class SwapTask:
    def __init__(self, cfg, rank, device):
        self.cfg = cfg
        self.rank = rank
        self.device = device

        # samples
        samples_path = os.path.join(cfg.run_dir, 'samples')
        self.samples_path = os.path.join(samples_path, 'step_{}_im_{}.png')
        os.makedirs(os.path.dirname(self.samples_path), exist_ok=True)

        # eval metrics
        self.eval_metrics_writer = MetricsWriter(
            cfg.run_dir,
            self._get_eval_metric_names(),
            'eval',
        )

        # val metrics
        self.val_metrics_writer = MetricsWriter(
            cfg.run_dir,
            self._get_val_metric_names(),
            'val',
        )


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
            snapshot_data['G'],
        ).eval().requires_grad_(False).to(self.device)

        # sampler
        sample_fn = trainer.get_sample_fn()

        # loop
        for batch in test_set:
            x_img_preds, y_img_preds, real_swaps = sample_fn(
                model,
                batch,
                batch_size,
                self.device,
            )
            x_img_preds = x_img_preds.cpu().numpy()
            y_img_preds = y_img_preds.cpu().numpy()
            real_swaps = real_swaps.cpu().numpy()
            for id, x_img_pred, y_img_pred, real_swap in zip(
                batch['item_id'],
                x_img_preds,
                y_img_preds,
                real_swaps,
            ):
                # TODO: generalize and move into helper function
                canvas = Image.new(
                    'RGB',
                    (self.cfg.dataset.imsize * 3, self.cfg.dataset.imsize),
                    'black',
                )
                canvas.paste(
                    Image.fromarray(
                        np.transpose(real_swap, (1, 2, 0)),
                        'RGB',
                    ),
                    (0, 0),
                )
                canvas.paste(
                    Image.fromarray(
                        np.transpose(x_img_pred, (1, 2, 0)),
                        'RGB',
                    ),
                    (self.cfg.dataset.imsize, 0),
                )
                canvas.paste(
                    Image.fromarray(
                        np.transpose(y_img_pred, (1, 2, 0)),
                        'RGB',
                    ),
                    (self.cfg.dataset.imsize * 2, 0),
                )
                canvas.save(self.samples_path.format(cur_step, id))


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
                snapshot_data['G'],
            ).eval().requires_grad_(False).to(self.device)
            face_loss_model = FaceLoss().eval().requires_grad_(False).to(
                self.device)
            lpips_loss_model = LpipsLoss().eval().requires_grad_(False).to(
                self.device)
            gender_loss_model = GenderLoss().eval().requires_grad_(False).to(
                self.device)

            # evaluator
            eval_fn = trainer.get_eval_fn()

            l2_pixel = 0.
            face_loss = 0.
            lpips_loss = 0.
            gender_loss = 0.
            for batch in val_set:
                x_img_pred, y_img_pred, real_swap = eval_fn(
                    model,
                    batch,
                    batch_size,
                    self.device,
                )

                x_img = batch['x_img'].to(self.device).to(torch.float32) \
                    / 127.5 - 1
                y_img = batch['y_img'].to(self.device).to(torch.float32) \
                    / 127.5 - 1
                real_img = batch['real_img'].to(self.device).to(torch.float32) \
                    / 127.5 - 1
                real_gender = batch['real_gender'].to(self.device).to(
                    torch.float32)

                l2_pixel += (F.mse_loss(x_img, x_img_pred).mean() + \
                    F.mse_loss(y_img, y_img_pred).mean()) / 2

                face_loss += (face_loss_model(x_img, x_img_pred).mean() + \
                    face_loss_model(y_img, y_img_pred).mean()) / 2

                lpips_loss += (lpips_loss_model(x_img, x_img_pred).mean() + \
                    lpips_loss_model(y_img, y_img_pred).mean()) / 2

                target_gender = -1. * real_gender + 1. # flip 0s and 1s
                gender_loss += gender_loss_model(real_swap, target_gender)

            l2_pixel /= len(val_set)
            face_loss /= len(val_set)
            lpips_loss /= len(val_set)
            gender_loss /= len(val_set)

            self.eval_metrics_writer.write_metric(
                cur_step,
                {
                    'l2_pixel': l2_pixel.item(),
                    'face': face_loss.item(),
                    'lpips': lpips_loss.item(),
                    'gender': gender_loss.item(),
                },
            )

            self.val_metrics_writer.write_metric(
                cur_step,
                {'todo': 0.},
            )


    def _get_eval_metric_names(self):
        return ['l2_pixel', 'face', 'lpips', 'gender']


    def _get_val_metric_names(self):
        return ['todo']
