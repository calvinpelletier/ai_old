#!/usr/bin/env python3
from ai_old.util.etc import print_
from ai_old.util.metrics_writer import MetricsWriter
from ai_old.util.factory import build_dataset
import copy
import torch
import torch.nn.functional as F
from ai_old.util.etc import binary_acc


class GenderClassificationTask:
    def __init__(self, cfg, rank, device):
        self.cfg = cfg
        self.rank = rank
        self.device = device

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
        return


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
                snapshot_data['C'],
            ).eval().requires_grad_(False).to(self.device)

            # evaluator
            eval_fn = trainer.get_eval_fn()

            # calc metrics
            bce = 0.
            acc = 0.
            for batch in val_set:
                pred, target = eval_fn(model, batch, batch_size, self.device)
                bce += F.binary_cross_entropy_with_logits(pred, target).mean()
                acc += binary_acc(pred, target).mean()
            bce /= len(val_set)
            acc /= len(val_set)

            self.eval_metrics_writer.write_metric(
                cur_step,
                {
                    'bce': bce.item(),
                    'acc': acc.item(),
                },
            )

            self.val_metrics_writer.write_metric(
                cur_step,
                {'todo': 0.},
            )


    def _get_eval_metric_names(self):
        return ['bce', 'acc']


    def _get_val_metric_names(self):
        return ['todo']
