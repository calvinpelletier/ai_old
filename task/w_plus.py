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
from ai_old.loss.perceptual.face import FaceIdLoss
from ai_old.loss.clip import GenderSwapClipLoss


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
        sample_imsize = 256
        for batch in test_set:
            swaps = sample_fn(
                model,
                batch,
                batch_size,
                self.device,
                sample_imsize=sample_imsize,
            )
            swaps = [swap.cpu().numpy() for swap in swaps]
            for i in range(len(batch['item_id'])):
                id = batch['item_id'][i]
                # TODO: generalize and move into helper function
                canvas = Image.new(
                    'RGB',
                    (sample_imsize * len(swaps), sample_imsize),
                    'black',
                )
                for j, swap in enumerate(swaps):
                    canvas.paste(
                        Image.fromarray(
                            np.transpose(swap[i], (1, 2, 0)),
                            'RGB',
                        ),
                        (sample_imsize * j, 0),
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
            # face_loss_model = FaceIdLoss(
            #     cfg.dataset.imsize).eval().requires_grad_(False).to(self.device)
            # clip_loss_model = GenderSwapClipLoss(
            #     cfg.dataset.imsize,
            #     cfg.loss.clip.female_male_target_texts,
            #     self.device,
            # ).eval().requires_grad_(False).to(self.device)
            face_loss_model = trainer.face_loss_model
            clip_loss_model = trainer.clip_loss_model

            # evaluator
            eval_fn = trainer.get_eval_fn()

            clip_loss = 0.
            face_loss = 0.
            delta_loss = 0.
            for batch in val_set:
                real_img, real_gender, swap_img, delta = eval_fn(
                    model,
                    batch,
                    batch_size,
                    self.device,
                )

                face_loss += face_loss_model(swap_img, real_img).mean()
                clip_loss += clip_loss_model(
                    real_img,
                    swap_img,
                    real_gender.clone().detach(),
                ).mean()
                delta_loss += delta.square().mean()

            clip_loss /= len(val_set)
            face_loss /= len(val_set)
            delta_loss /= len(val_set)
            unweighted_total = clip_loss + face_loss + delta_loss

            self.eval_metrics_writer.write_metric(
                cur_step,
                {
                    'clip': clip_loss.item(),
                    'face': face_loss.item(),
                    'delta': delta_loss.item(),
                    'unweighted_total': unweighted_total.item(),
                },
            )

            self.val_metrics_writer.write_metric(
                cur_step,
                {'todo': 0.},
            )


    def _get_eval_metric_names(self):
        return ['clip', 'face', 'delta', 'unweighted_total']


    def _get_val_metric_names(self):
        return ['todo']
