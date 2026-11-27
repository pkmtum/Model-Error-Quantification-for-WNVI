import torch
from model.Distributions.mean_functions.base_mean_function import MeanFunction


class ConstantScalarMean(MeanFunction):
    def __init__(self, value, dim):
        super().__init__(value=value, dim=dim)
        self.value = float(value)

    def compute(self, num_samples=None, **kwargs):
        if num_samples is not None:
            shape = (num_samples, self.dim)
        else:
            shape = (1, self.dim)
        return torch.full(shape, self.value)
