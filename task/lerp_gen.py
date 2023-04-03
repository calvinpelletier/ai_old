#!/usr/bin/env python3
from ai_old.util.etc import print_
from ai_old.util.factory import build_dataset
import copy
from ai_old.util.etc import normalized_tensor_to_pil_img, create_img_row
from ai_old.task.lerp import WPlusLerpTask


BATCH_SIZE = 16


class LerpGenTask(WPlusLerpTask):
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
            snapshot_data[trainer.get_model_key_for_eval()],
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
            for i, id in enumerate(batch['item_id']):
                row = [
                    normalized_tensor_to_pil_img(imgs[j][i]) \
                    for j in range(4)
                ]
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
                snapshot_data[trainer.get_model_key_for_eval()],
            ).eval().requires_grad_(False).to(self.device)

            # evaluator
            eval_fn = trainer.get_eval_fn()

            avg_face_loss = 0.
            avg_delta_loss = 0.
            avg_classify_loss = 0.
            avg_reg_loss = 0.
            for batch in val_set:
                delta_loss, face_loss, classify_loss, reg_loss = eval_fn(
                    model,
                    batch,
                    BATCH_SIZE,
                    self.device,
                )

                avg_face_loss += face_loss
                avg_delta_loss += delta_loss
                avg_classify_loss += classify_loss
                avg_reg_loss += reg_loss

            avg_face_loss /= len(val_set)
            avg_delta_loss /= len(val_set)
            avg_classify_loss /= len(val_set)
            avg_reg_loss /= len(val_set)
            unweighted_total = avg_face_loss + avg_delta_loss + \
                avg_classify_loss  + avg_reg_loss

            self.eval_metrics_writer.write_metric(
                cur_step,
                {
                    'face': avg_face_loss.item(),
                    'delta': avg_delta_loss.item(),
                    'classify': avg_classify_loss.item(),
                    'reg': avg_reg_loss.item(),
                    'unweighted_total': unweighted_total.item(),
                },
            )

            self.val_metrics_writer.write_metric(
                cur_step,
                {'todo': 0.},
            )


    def _get_eval_metric_names(self):
        return [
            'face',
            'delta',
            'classify',
            'reg',
            'unweighted_total',
        ]


    def _get_val_metric_names(self):
        return ['todo']
