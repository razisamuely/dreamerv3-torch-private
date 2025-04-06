import sys
import yaml
import pathlib
import yaml
import pathlib
import types
import sys


def load_config():
    # Load YAML file
    config_path = pathlib.Path(sys.argv[0]).parent.parent / "configs.yaml"
    config_data = yaml.safe_load(config_path.read_text())
    
    # Get defaults section
    defaults = config_data["defaults"]
    
    # Helper function to convert values and handle nested dictionaries
    def process_value(value):
        if isinstance(value, dict):
            # Keep as dictionary for compatibility with ** unpacking
            return {k: process_value(v) for k, v in value.items()}
        elif isinstance(value, str):
            # Try to convert scientific notation strings to float
            if 'e' in value.lower() or 'e+' in value.lower() or 'e-' in value.lower():
                try:
                    return float(value)
                except ValueError:
                    pass
        return value
    
    # Create config object with proper types
    config = types.SimpleNamespace()
    for key, value in defaults.items():
        setattr(config, key, process_value(value))
    
    # Add additional settings
    config.logdir = str(pathlib.Path(__file__).parent.parent / "logs")
    config.traindir = config.traindir or pathlib.Path(config.logdir) / "train_eps"
    
    # Set multi-agent configuration
    config.n_agents = 4  # Set number of agents for testing
    
    return config