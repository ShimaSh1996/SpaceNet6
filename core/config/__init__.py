import os
import yaml
from .dotmap import DotMap

cfg = None

__location__ = os.path.abspath(os.path.dirname(__file__))
BASE_CFG_PATH = os.path.join(__location__, "base.yaml")

with open(BASE_CFG_PATH) as fh:
    cfg_dict = yaml.unsafe_load(fh)
cfg = DotMap(cfg_dict)