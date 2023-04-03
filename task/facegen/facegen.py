#!/usr/bin/env python3
from ai_old.task.manifold import ImgManifoldTask


class FacegenTask(ImgManifoldTask):
    def _finish_eval(self, val_dataset_len):
        self._calc_manifold_metrics()

        # write results to disk
        self.eval_metrics_writer.write_metric(
            self.total_steps,
            self.eval_metrics,
        )
