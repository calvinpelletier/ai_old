#!/usr/bin/env python3
from ai_old.util.etc import print_
from PIL import Image
from ai_old.util.factory import build_dataset
import copy
import numpy as np
import torch
import torch.nn.functional as F
from ai_old.task.rec import RecTask
from ai_old.util.etc import create_img_row, normalized_tensor_to_pil_img


class EncLerpTask(RecTask):
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
            imgs0, imgs1, imgs2, imgs3 = sample_fn(
                model,
                batch,
                batch_size,
                self.device,
            )
            for id, img0, img1, img2, img3 in zip(
                batch['item_id'],
                imgs0,
                imgs1,
                imgs2,
                imgs3,
            ):
                create_img_row(
                    [
                        normalized_tensor_to_pil_img(img0),
                        normalized_tensor_to_pil_img(img1),
                        normalized_tensor_to_pil_img(img2),
                        normalized_tensor_to_pil_img(img3),
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

            # evaluator
            eval_fn = trainer.get_eval_fn()

            # calc metrics
            l2_enc = 0.
            for batch in val_set:
                pred_enc, target_enc = eval_fn(
                    model, batch, batch_size, self.device)
                l2_enc += F.mse_loss(pred_enc, target_enc)
                # l2_enc += torch.mean(torch.abs(pred_enc - target_enc))
            l2_enc /= len(val_set)

            self.eval_metrics_writer.write_metric(
                cur_step,
                {
                    'l2_enc': l2_enc.item(),
                },
            )

            self.val_metrics_writer.write_metric(
                cur_step,
                {'todo': 0.},
            )


    def _get_eval_metric_names(self):
        return ['l2_enc']


    def _get_val_metric_names(self):
        return ['todo']
