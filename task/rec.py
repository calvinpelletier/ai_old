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
from ai_old.loss.perceptual.face import FaceIdLoss
import lpips


class RecTask:
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
            snapshot_data[trainer.get_model_key_for_eval()],
        ).eval().requires_grad_(False).to(self.device)

        # sampler
        sample_fn = trainer.get_sample_fn()

        # loop
        for batch in test_set:
            imgs = sample_fn(
                model,
                batch,
                batch_size,
                self.device,
            ).cpu().numpy()
            for id, generated in zip(batch['item_id'], imgs):
                Image.fromarray(
                    np.transpose(generated, (1, 2, 0)),
                    'RGB',
                ).save(self.samples_path.format(cur_step, id))

        # debug
        # avg_img = model.g(
        #     model.e.w_avg.unsqueeze(dim=0).repeat(1, model.e.num_ws, 1),
        # )[0]
        # avg_img = (avg_img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        # avg_img = avg_img.cpu().numpy()
        # Image.fromarray(
        #     np.transpose(avg_img, (1, 2, 0)),
        #     'RGB',
        # ).save('/home/asiu/data/tmp/avg_img.png')


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
            face_loss_model = FaceIdLoss(model.imsize).eval().requires_grad_(
                False).to(self.device)
            lpips_loss_model = lpips.LPIPS(net='alex').eval().requires_grad_(
                False).to(self.device)

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
