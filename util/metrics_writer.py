import os


class MetricsWriter(object):
    def __init__(self, output_dir, metric_names, fname):
        self.output_dir = output_dir
        self.metric_names = metric_names

        self.eval_csv_path = os.path.join(output_dir, f'{fname}.csv')

        with open(self.eval_csv_path, 'a') as fw:
            fw.write(','.join(['step'] + metric_names) + '\n')

    def write_metric(self, step, metrics):
        str_values = [str(step)] + [
            self.__metric_to_string(m, metrics[m]) for m in self.metric_names
        ]
        with open(self.eval_csv_path, 'a') as fw:
            fw.write(','.join(str_values) + '\n')

        print(f'[EVAL][{str_values[0]}] ' + ','.join(
            [f'{k}={v}' for k, v in zip(self.metric_names, str_values[1:])]
        ))

    def __metric_to_string(self, metric_name, metric_value):
        # We can probably come back later and make this more interesting if needed.
        if metric_value is None:
            return 'null'
        elif isinstance(metric_value, float):
            return '%.4f' % metric_value
        else:
            return str(metric_value)
