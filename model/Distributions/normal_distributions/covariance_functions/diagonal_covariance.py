import torch
import torch.nn as nn
from model.Distributions.normal_distributions.base_covariance_function import (
    CovarianceFunction,
)
from utils.torch_funcs.function_make_torch_tensor import make_torch_tensor


class DiagonalCovariance(CovarianceFunction):
    def __init__(self, dim, std=1.0):
        super().__init__(dim=dim, std=std)
        log_std = torch.log(torch.tensor(std))
        self.log_std = nn.Parameter(make_torch_tensor(log_std, torch.Size([dim])))

    def compute(self, num_samples=None, **kwargs):
        std = self.log_std.exp()
        cov = torch.diag_embed(std**2)
        if num_samples is not None:
            if isinstance(num_samples, (list, tuple, torch.Size)):
                shape = tuple(num_samples) + cov.shape
            else:
                shape = (num_samples,) + cov.shape
            cov = cov.expand(*shape)
        return cov

    def sample(self, mean, num_samples=None, **kwargs):
        std = self.log_std.exp()
        shape = self.get_sample_shape(num_samples, mean.shape)
        return mean + torch.randn(*shape, device=mean.device) * std

    def log_prob(self, x, mean, num_samples=None, **kwargs):
        std = self.log_std.exp()
        var = std**2
        # Broadcast mean and std to x if needed
        log_term = torch.log(2 * torch.pi * var)
        quad_term = ((x - mean) ** 2) / var
        return -0.5 * (log_term + quad_term).sum(dim=-1)

    def entropy(self, **kwargs):
        std = self.log_std.exp()
        var = std**2
        # H = 0.5 * sum(log(2 * pi * e * var))
        return 0.5 * torch.sum(
            torch.log(
                2 * torch.pi * torch.exp(torch.tensor(1.0, device=std.device)) * var
            )
        )
