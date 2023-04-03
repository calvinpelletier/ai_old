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
from ai_old.loss.perceptual.face import FaceIdLoss
from ai_old.loss.ss import SynthSwapDeltaLoss
from ai_old.loss.clip import MultiTextClipLoss
from ai_old.util.etc import normalized_tensor_to_pil_img, create_img_row
from ai_old.loss.hair import calc_nonhair_l2_pixel_loss
from ai_old.loss.gender import GenderLoss


BATCH_SIZE = 16


class WPlusLerpTask:
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
            )
            imgs = imgs.cpu().numpy()
            for id, img in zip(batch['item_id'], imgs):
                Image.fromarray(
                    np.transpose(img, (1, 2, 0)),
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

            face_loss_model = trainer.face_loss_model
            if face_loss_model is None:
                face_loss_model = FaceIdLoss(
                    self.cfg.dataset.imsize,
                ).eval().requires_grad_(False).to(self.device)

            clip_loss_model = MultiTextClipLoss(
                {
                    'main': ['male face', 'female face'],
                    'hair': ['short hair', 'long hair'],
                },
                self.device,
            ).eval().requires_grad_(False).to(self.device)

            ss_delta_loss_model = trainer.ss_delta_loss_model
            if ss_delta_loss_model is None:
                ss_delta_loss_model = \
                    SynthSwapDeltaLoss().eval().requires_grad_(False).to(
                        self.device)

            classify_loss_model = trainer.classify_loss_model
            if classify_loss_model is None:
                classify_loss_model = GenderLoss().eval().requires_grad_(
                    False).to(self.device)

            # evaluator
            eval_fn = trainer.get_eval_fn()

            clip_loss = 0.
            face_loss = 0.
            delta_loss = 0.
            ss_delta_loss = 0.
            hair_clip_loss = 0.
            classify_loss = 0.
            for batch in val_set:
                img, gender, w, new_img, new_w = eval_fn(
                    model,
                    batch,
                    BATCH_SIZE,
                    self.device,
                )

                face_loss += face_loss_model(new_img, img).mean()
                delta_loss += F.mse_loss(new_w, w)
                ss_delta_loss += ss_delta_loss_model(new_w, w, gender)
                clip_losses = clip_loss_model('all', new_img, gender)
                clip_loss += clip_losses['main']
                hair_clip_loss += clip_losses['hair']
                classify_loss += classify_loss_model(new_img, 1. - gender)

            clip_loss /= len(val_set)
            face_loss /= len(val_set)
            delta_loss /= len(val_set)
            ss_delta_loss /= len(val_set)
            hair_clip_loss /= len(val_set)
            classify_loss /= len(val_set)
            unweighted_total = clip_loss + face_loss + delta_loss + \
                ss_delta_loss + hair_clip_loss + classify_loss

            self.eval_metrics_writer.write_metric(
                cur_step,
                {
                    'clip': clip_loss.item(),
                    'face': face_loss.item(),
                    'delta': delta_loss.item(),
                    'ss_delta': ss_delta_loss.item(),
                    'hair_clip': hair_clip_loss.item(),
                    'classify': classify_loss.item(),
                    'unweighted_total': unweighted_total.item(),
                },
            )

            self.val_metrics_writer.write_metric(
                cur_step,
                {'todo': 0.},
            )


    def _get_eval_metric_names(self):
        return [
            'clip',
            'face',
            'delta',
            'ss_delta',
            'hair_clip',
            'classify',
            'unweighted_total',
        ]


    def _get_val_metric_names(self):
        return ['todo']


class WPlusDualLerpTask(WPlusLerpTask):
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
            imgs1, imgs2 = sample_fn(
                model,
                batch,
                BATCH_SIZE,
                self.device,
            )
            for id, img1, img2 in zip(batch['item_id'], imgs1, imgs2):
                row = [normalized_tensor_to_pil_img(x) for x in [img1, img2]]
                create_img_row(row, self.cfg.dataset.imsize).save(
                    self.samples_path.format(cur_step, id))


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

            face_loss_model = trainer.face_loss_model
            if face_loss_model is None:
                face_loss_model = FaceIdLoss(
                    self.cfg.dataset.imsize,
                ).eval().requires_grad_(False).to(self.device)

            clip_loss_model = MultiTextClipLoss(
                {
                    'main': ['male face', 'female face'],
                    'hair': ['short hair', 'long hair'],
                },
                self.device,
            ).eval().requires_grad_(False).to(self.device)

            ss_delta_loss_model = trainer.ss_delta_loss_model
            if ss_delta_loss_model is None:
                ss_delta_loss_model = \
                    SynthSwapDeltaLoss().eval().requires_grad_(False).to(
                        self.device)

            # evaluator
            eval_fn = trainer.get_eval_fn()

            clip_loss = 0.
            face_loss = 0.
            delta_loss = 0.
            ss_delta_loss = 0.
            hair_clip_loss = 0.
            nonhair_l2_pixel_loss = 0.
            for batch in val_set:
                img, gender, w, w1, w2, img1, img2, seg1, seg2 = eval_fn(
                    model,
                    batch,
                    BATCH_SIZE,
                    self.device,
                )

                face_loss += face_loss_model(img2, img1).mean()
                delta_loss += F.mse_loss(w2, w1)
                ss_delta_loss += ss_delta_loss_model(w2, w1, gender)
                clip_losses = clip_loss_model('all', img2, gender)
                clip_loss += clip_losses['main']
                hair_clip_loss += clip_losses['hair']
                nonhair_l2_pixel_loss += calc_nonhair_l2_pixel_loss(
                    img1, img2, seg1, seg2)

            clip_loss /= len(val_set)
            face_loss /= len(val_set)
            delta_loss /= len(val_set)
            ss_delta_loss /= len(val_set)
            hair_clip_loss /= len(val_set)
            nonhair_l2_pixel_loss /= len(val_set)
            unweighted_total = clip_loss + face_loss + delta_loss + \
                ss_delta_loss + hair_clip_loss + nonhair_l2_pixel_loss

            self.eval_metrics_writer.write_metric(
                cur_step,
                {
                    'clip': clip_loss.item(),
                    'face': face_loss.item(),
                    'delta': delta_loss.item(),
                    'ss_delta': ss_delta_loss.item(),
                    'hair_clip': hair_clip_loss.item(),
                    'nonhair_l2_pixel': nonhair_l2_pixel_loss.item(),
                    'unweighted_total': unweighted_total.item(),
                },
            )

            self.val_metrics_writer.write_metric(
                cur_step,
                {'todo': 0.},
            )


    def _get_eval_metric_names(self):
        return [
            'clip',
            'face',
            'delta',
            'ss_delta',
            'hair_clip',
            'nonhair_l2_pixel',
            'unweighted_total',
        ]


    def _get_val_metric_names(self):
        return ['todo']
