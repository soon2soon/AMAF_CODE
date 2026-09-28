from pathlib import Path
import random
import numpy as np
import torch
from amaf.config import load_config_with_base
from amaf.utils.seed import seed_everything
ROOT=Path(__file__).resolve().parents[1]

def test_config_inheritance():
    c=load_config_with_base(ROOT/"configs/lunarlander/amaf_dqn.yaml")
    assert c["algorithm"]=="amaf_dqn" and c["network"]["n_heads"]==4 and c["target_mode"]=="double_dqn"

def test_seed_reproducibility():
    seed_everything(123); a=(random.random(),np.random.rand(),torch.rand(1).item())
    seed_everything(123); b=(random.random(),np.random.rand(),torch.rand(1).item())
    assert a==b
