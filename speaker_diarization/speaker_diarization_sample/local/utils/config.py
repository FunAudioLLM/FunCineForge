import os
import yaml

class Config(object):
    def __init__(self, conf_dict):
        for key, value in conf_dict.items():
            self.__dict__[key] = value


def convert_to_yaml(overrides):
    """Convert args to yaml for overrides"""
    yaml_string = ""

    # Handle '--arg=val' type args
    joined_args = "=".join(overrides)
    split_args = joined_args.split("=")

    for arg in split_args:
        if arg.startswith("--"):
            yaml_string += "\n" + arg[len("--") :] + ":"
        else:
            yaml_string += " " + arg

    return yaml_string.strip()


def yaml_config_loader(conf_file, overrides=None):
    with open(conf_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if overrides is not None:
        config.update(yaml.load(overrides, Loader=yaml.FullLoader))

    variables = {k: v for k, v in config.items() if isinstance(k, str) and not k.startswith('_') and isinstance(v, (int, float, str, bool))}

    def resolve(x):
        if isinstance(x, dict):
            return {k: resolve(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [resolve(item) for item in x]
        elif isinstance(x, str) and x.startswith('<') and x.endswith('>'):
            key = x[1:-1]
            return variables.get(key, x)
        else:
            return x
    return resolve(config)


def build_config(config_file, overrides=None, copy=False):
    if config_file.endswith(".yaml"):
        if overrides is not None:
            overrides = convert_to_yaml(overrides)
        conf_dict = yaml_config_loader(config_file, overrides)
        if copy and 'exp_dir' in conf_dict:
            os.makedirs(conf_dict['exp_dir'], exist_ok=True)
            saved_path = os.path.join(conf_dict['exp_dir'], 'config.yaml')
            with open(saved_path, 'w') as f:
                f.write(yaml.dump(conf_dict))
    else:
        raise ValueError("Unknown config file format")

    return Config(conf_dict)
