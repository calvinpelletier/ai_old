#!/usr/bin/env python3
import ai_old.constants as c
from ai_old.util.etc import print_
import os
from ai_old.util.metrics_writer import MetricsWriter
from PIL import Image
from ai_old.util.factory import build_dataset
import copy
import numpy as np
import torch
import torch.nn.functional as F


BATCH_SIZE = 8


# TODO: FID?
class OutpaintTask:
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

        # build the test dataset
        dataset_core = build_dataset(self.cfg.dataset)
        test_set = dataset_core.get_test_set(
            BATCH_SIZE,
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
            imgs = sample_fn(
                model,
                batch,
                BATCH_SIZE,
                self.device,
            ).cpu().numpy()
            for id, generated in zip(batch['item_id'], imgs):
                Image.fromarray(
                    np.transpose(generated, (1, 2, 0)),
                    'RGB',
                ).save(self.samples_path.format(cur_step, id))


    def eval(self, trainer, snapshot_data, cur_step):
        print_(self.rank, '[INFO] calculating metrics...')

        if self.rank == 0:
            # build the val dataset
            dataset_core = build_dataset(self.cfg.dataset)
            val_set = dataset_core.get_val_set(
                BATCH_SIZE,
                0, # seed
                self.rank,
                self.cfg.num_gpus,
                verbose=False,
            )

            # setup models
            model = copy.deepcopy(
                snapshot_data['G'],
            ).eval().requires_grad_(False).to(self.device)

            # evaluator
            eval_fn = trainer.get_eval_fn()

            loss = 0.
            for batch in val_set:
                gt, pred = eval_fn(
                    model,
                    batch,
                    BATCH_SIZE,
                    self.device,
                )
                loss += F.mse_loss(pred, gt).mean()
            loss /= len(val_set)

            self.eval_metrics_writer.write_metric(
                cur_step,
                {
                    'l2_pixel': loss.item(),
                },
            )

            self.val_metrics_writer.write_metric(
                cur_step,
                {'todo': 0.},
            )


    def _get_eval_metric_names(self):
        return ['l2_pixel']


    def _get_val_metric_names(self):
        return ['todo']
