import torch
import numpy as np
from utils.torch_funcs.function_cuda_to_cpu import cuda_to_cpu


class DynamicLogger:
    def __init__(self, track_vars):
        self.track_vars = track_vars
        self.history = {}

    def _process_value(self, value):
        # Always clone, detach, and move to CPU if tensor
        if isinstance(value, torch.Tensor):
            value = value.clone().detach()
            value = cuda_to_cpu(value)
            if value.numel() > 1:
                # Not scalar: save norm
                return torch.linalg.norm(value)
            else:
                return value.item() if value.numel() == 1 else value
        elif isinstance(value, dict):
            # Recursively process dicts
            return {k: self._process_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            # Recursively process lists/tuples
            return type(value)(self._process_value(v) for v in value)
        else:
            return value

    def log(self, var_space):
        for var in self.track_vars:
            value = var_space[var]
            processed = self._process_value(value)
            # Handle dicts: append to subdicts
            if isinstance(processed, dict):
                if var not in self.history:
                    self.history[var] = {k: [] for k in processed}
                for k, v in processed.items():
                    self.history[var][k].append(v)
            else:
                if var not in self.history:
                    self.history[var] = []
                self.history[var].append(processed)

    def save(self, path):
        torch.save(self.history, path + "/dynamic_logger.pt")