import yaml
from omegaconf import DictConfig

def read_yaml(path_to_yaml: str):
    with open(path_to_yaml, 'r') as f:
        return DictConfig(yaml.load(f.read(), Loader=yaml.FullLoader))