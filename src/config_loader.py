from types import SimpleNamespace
import yaml

def load_config(path: str) -> SimpleNamespace:
    """Loads a YAML file and returns a namespace object."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return _to_namespace(data)

def _to_namespace(data):
    """
    Recursively converts a yaml config to a namespace object.
    We wanna keep param_grid config as is, so it's faster
    """
    if isinstance(data, dict):
        return SimpleNamespace(**{k: (v if k == "param_grid" else _to_namespace(v)) for k, v in data.items()})
    elif isinstance(data, list):
        return [_to_namespace(item) for item in data]
    return data