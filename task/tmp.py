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


class TmpFacegenTask:
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
            snapshot_data['G_ema'],
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


    def eval(self, trainer, snapshot_data, cur_step):
        print_(self.rank, '[INFO] calculating metrics...')

        fid = metric_main.calc_metric(
            cfg=self.cfg,
            metric='fid50k_full',
            model=snapshot_data['G_ema'],
            eval_fn=trainer.get_eval_fn(),
            dataset_kwargs={
                'class_name': 'external.sg2.dataset.ImageFolderDataset',
                'path': self._get_img_folder_for_manifold_calc(),
                'resolution': self.cfg.dataset.imsize,
                'max_size': None,
                'use_labels': False,
                'xflip': False,
            },
            num_gpus=self.cfg.num_gpus,
            rank=self.rank,
            device=self.device,
        )

        if self.rank == 0:
            fid_calc_time = fid['total_time']
            print(f'fid calc time: {fid_calc_time:.2f}')

            self.eval_metrics_writer.write_metric(
                cur_step,
                {'fid': fid['results']['fid50k_full']},
            )

            self.val_metrics_writer.write_metric(
                cur_step,
                {'todo': 0.},
            )


    def _get_eval_metric_names(self):
        return ['fid']


    def _get_val_metric_names(self):
        return ['todo']


    def _get_img_folder_for_manifold_calc(self):
        if self.cfg.dataset.imsize == 128:
            return os.path.join(c.ASI_DATASETS_PATH, 'ffhq-128/x')
        elif self.cfg.dataset.imsize == 256:
            return os.path.join(
                c.ASI_DATASETS_PATH,
                c.SUPPLEMENTAL_DATASET_FOLDER_NAME,
                'face_image_256/ffhq-128',
            )
        else:
            raise Exception(str(self.cfg.dataset.imsize))


class TmpInversionTask:
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

        # setup model
        model = copy.deepcopy(
            snapshot_data['G'],
        ).eval().requires_grad_(False).to(self.device)

        # build the test dataset
        batch_size = 32
        dataset_core = build_dataset(self.cfg.dataset)
        test_set = dataset_core.get_test_set(
            batch_size,
            0, # seed
            self.rank,
            self.cfg.num_gpus,
            verbose=False,
        )

        # sampler
        sample_fn = trainer.get_sample_fn()

        # loop
        for batch in test_set:
            gen_imgs, rec_imgs = sample_fn(
                model,
                batch,
                batch_size,
                self.device,
            )
            gen_imgs = gen_imgs.cpu().numpy()
            rec_imgs = rec_imgs.cpu().numpy()
            for id, gen, rec in zip(batch['item_id'], gen_imgs, rec_imgs):
                # TODO: generalize and move into helper function
                canvas = Image.new(
                    'RGB',
                    (self.cfg.dataset.imsize * 2, self.cfg.dataset.imsize),
                    'black',
                )
                canvas.paste(
                    Image.fromarray(
                        np.transpose(gen, (1, 2, 0)),
                        'RGB',
                    ),
                    (0, 0),
                )
                canvas.paste(
                    Image.fromarray(
                        np.transpose(rec, (1, 2, 0)),
                        'RGB',
                    ),
                    (self.cfg.dataset.imsize, 0),
                )
                canvas.save(self.samples_path.format(cur_step, id))


    def eval(self, trainer, snapshot_data, cur_step):
        print_(self.rank, '[INFO] calculating metrics...')

        if self.rank == 0:
            # setup model
            model = copy.deepcopy(
                snapshot_data['G'],
            ).eval().requires_grad_(False).to(self.device)

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

            # evaluator
            eval_fn = trainer.get_eval_fn()

            # loop
            l2_pixel = 0.
            for batch in val_set:
                gen_img, rec_img = eval_fn(
                    model,
                    batch,
                    batch_size,
                    self.device,
                )
                l2_pixel += F.mse_loss(gen_img, rec_img).mean()
            l2_pixel /= len(val_set)

            self.eval_metrics_writer.write_metric(
                cur_step,
                {'l2_pixel': l2_pixel.item()},
            )

            self.val_metrics_writer.write_metric(
                cur_step,
                {'todo': 0.},
            )


    def _get_eval_metric_names(self):
        return ['l2_pixel']


    def _get_val_metric_names(self):
        return ['todo']



class TmpRecTask:
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


    def eval(self, trainer, snapshot_data, cur_step):
        print_(self.rank, '[INFO] calculating metrics...')

        fid = metric_main.calc_metric(
            cfg=self.cfg,
            metric='fid50k_full',
            model=snapshot_data['G'],
            eval_fn=trainer.get_eval_fn(),
            dataset_kwargs={
                'class_name': 'external.sg2.dataset.ImageFolderDataset',
                'path': self._get_img_folder_for_manifold_calc(),
                'resolution': self.cfg.dataset.imsize,
                'max_size': None,
                'use_labels': False,
                'xflip': False,
            },
            num_gpus=self.cfg.num_gpus,
            rank=self.rank,
            device=self.device,
        )

        if self.rank == 0:
            fid_calc_time = fid['total_time']
            print(f'fid calc time: {fid_calc_time:.2f}')

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

            # setup model
            model = copy.deepcopy(
                snapshot_data['G'],
            ).eval().requires_grad_(False).to(self.device)

            # evaluator
            eval_fn = trainer.get_eval_fn()

            l2_pixel = 0.
            for batch in val_set:
                rec = eval_fn(
                    model,
                    batch,
                    batch_size,
                    self.device,
                )
                l2_pixel += F.mse_loss(
                    batch['y'].to(self.device).to(torch.float32) / 127.5 - 1,
                    rec.to(torch.float32) / 127.5 - 1,
                ).mean()
            l2_pixel /= len(val_set)

            self.eval_metrics_writer.write_metric(
                cur_step,
                {
                    'fid': fid['results']['fid50k_full'],
                    'l2_pixel': l2_pixel,
                },
            )

            self.val_metrics_writer.write_metric(
                cur_step,
                {'todo': 0.},
            )


    def _get_eval_metric_names(self):
        return ['fid']


    def _get_val_metric_names(self):
        return ['todo']


    def _get_img_folder_for_manifold_calc(self):
        assert self.cfg.dataset.imsize == 128
        return os.path.join(c.ASI_DATASETS_PATH, 'ffhq-128/x')


class TmpFgFacegenTask(TmpFacegenTask):
    def _get_img_folder_for_manifold_calc(self):
        assert self.cfg.dataset.imsize == 128
        return os.path.join(
            c.ASI_DATASETS_PATH,
            c.SUPPLEMENTAL_DATASET_FOLDER_NAME,
            'fg',
            'ffhq-128',
        )
