import torch
import torch.nn as nn
from model.Distributions.mean_functions.base_mean_function import (
    MeanFunction,
)
from utils.torch_funcs.function_make_torch_tensor import make_torch_tensor


class LearnableMean(MeanFunction):
    def __init__(self, dim, value: float | list[float] | torch.Tensor | str = 0.0):
        super().__init__(value=value, dim=dim)
        if isinstance(value, str) and value == "random":
            value = torch.randn(dim)
        self.mean = nn.Parameter(make_torch_tensor(value, torch.Size([dim])))

    def compute(self, num_samples=None, **kwargs):
        if num_samples is not None:
            shape = (num_samples, self.mean.numel())
        else:
            shape = (1, self.mean.numel())
        return self.mean.expand(*shape)
