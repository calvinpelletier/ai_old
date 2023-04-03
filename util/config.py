import yaml


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                self.__dict__[key] = AttrDict(value)
            elif isinstance(value, (list, tuple)):
                if len(value) > 0 and isinstance(value[0], dict):
                    self.__dict__[key] = [AttrDict(item) for item in value]
                else:
                    self.__dict__[key] = value


def load(path):
    with open(path, 'r') as f:
        conf = yaml.safe_load(f)
    return AttrDict(**conf)


def get_default(conf, attr, default):
    if hasattr(conf, attr):
        return getattr(conf, attr)
    return default
