import torch
from model.Distributions.mean_functions.base_mean_function import MeanFunction


class ConstantVectorMean(MeanFunction):
    def __init__(self, value, dim):
        super().__init__(value=value, dim=dim)
        self.value = torch.as_tensor(value, dtype=torch.float32)
        if self.value.shape != (self.dim,):
            if self.value.numel() == 1:
                self.value = self.value.expand(self.dim)
            else:
                raise ValueError(
                    f"Value shape {self.value.shape} does not match dim {self.dim}"
                )

    def compute(self, num_samples=None, **kwargs):
        if num_samples is not None:
            shape = (num_samples, self.dim)
        else:
            shape = (1, self.dim)
        return self.value.expand(*shape)
