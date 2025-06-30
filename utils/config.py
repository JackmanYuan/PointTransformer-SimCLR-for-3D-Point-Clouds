# utils/config.py
from types import SimpleNamespace

def get_default_cfg():
    cfg = SimpleNamespace()
    cfg.input_dim = 6  # xyz + rgb
    cfg.num_point = 2048
    cfg.num_class = 40  # not used, just placeholder
    cfg.batch_size = 16
    cfg.epochs = 100  # <-- Add this line

    cfg.model = SimpleNamespace()
    cfg.model.nblocks = 3
    cfg.model.nneighbor = 16
    cfg.model.transformer_dim = 64

    return cfg