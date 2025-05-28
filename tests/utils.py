import sys
import yaml
import pathlib
import types

# Constants
CONFIG_FILENAME = "configs.yaml"
LOGS_DIR = "logs"
TRAIN_DIR = "train_eps"
DEFAULT_AGENTS = 4
SCIENTIFIC_NOTATION_INDICATORS = ['e', 'e+', 'e-']


def load_config():
    config_path = pathlib.Path(sys.argv[0]).parent.parent / CONFIG_FILENAME
    config_data = yaml.safe_load(config_path.read_text())
    defaults = config_data["defaults"]
    
    config = types.SimpleNamespace()
    _populate_config_attributes(config, defaults)
    
    _add_path_settings(config)
    config.n_agents = DEFAULT_AGENTS
    
    return config


def _populate_config_attributes(config, defaults):
    for key, value in defaults.items():
        setattr(config, key, _process_value(value))


def _process_value(value):
    if isinstance(value, dict):
        return {k: _process_value(v) for k, v in value.items()}
    elif isinstance(value, str) and _is_scientific_notation(value):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def _is_scientific_notation(value):
    return any(indicator in value.lower() for indicator in SCIENTIFIC_NOTATION_INDICATORS)


def _add_path_settings(config):
    config.logdir = str(pathlib.Path(__file__).parent.parent / LOGS_DIR)
    config.traindir = config.traindir or pathlib.Path(config.logdir) / TRAIN_DIR