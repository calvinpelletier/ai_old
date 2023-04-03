#!/usr/bin/env python3
from ai_old.task.manifold import ImgManifoldTask


class LightFemaleFacegenTask(ImgManifoldTask):
    def _finish_eval(self, val_dataset_len):
        self._calc_manifold_metrics()

        # write results to disk
        self.eval_metrics_writer.write_metric(
            self.total_steps,
            self.eval_metrics,
        )

    def _get_min_steps_slow_metrics(self):
        return 10000

    def _get_n_ephemeral(self):
        return 5000
