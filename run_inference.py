"""Run the full two-stage brain experiment.

Usage: python run_inference.py configs/brain_35dB.py
"""
import argparse
import copy
import importlib.util
import os
import random
import runpy
import types

import numpy as np
import torch

from utils.function_copy_input_file import copy_input_file
from utils.function_create_result_dict import create_result_dict

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("config", help="Path to config, e.g. configs/brain_35dB.py")
parser.add_argument("--seed", type=int, default=None, help="Override config seed")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location("config", args.config)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

seed = args.seed if args.seed is not None else getattr(cfg, "seed", 0)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

result_path = create_result_dict(dir_name=None)
copy_input_file(args.config, result_path)

stages = getattr(cfg, "stages", [dict(name="stage1")])
base = {k: v for k, v in vars(cfg).items() if not k.startswith("__") and not isinstance(v, types.ModuleType)}

init_state, init_u_hat = None, None
for i, overrides in enumerate(stages):
    overrides = dict(overrides)
    name = overrides.pop("name", f"stage{i + 1}")
    print(f"\n######## {name} ########")
    inp = types.SimpleNamespace(**copy.deepcopy({**base, **overrides}), torch=torch)
    stage_path = os.path.join(result_path, name)
    os.makedirs(stage_path)
    out = runpy.run_path(
        "stage_runner.py",
        init_globals=dict(
            inp=inp,
            result_path=stage_path,
            init_state=init_state,
            init_u_hat=init_u_hat,
        ),
    )
    init_state, init_u_hat = out["final_state"], out["final_u_hat"]
