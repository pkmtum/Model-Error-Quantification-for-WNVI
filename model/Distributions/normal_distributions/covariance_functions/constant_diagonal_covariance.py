import torch
from model.Distributions.normal_distributions.base_covariance_function import (
    CovarianceFunction,
)


class ConstantDiagonalCovariance(CovarianceFunction):
    def __init__(self, std, dim=1):
        """
        std: tensor or float, shape (dim,) or scalar
        """
        super().__init__(dim=dim, std=std)
        self.register_buffer("std", torch.as_tensor(std).float())

    def compute(self, num_samples=None, **kwargs):
        std = self.std
        cov = torch.diag_embed(std**2)
        if num_samples is not None:
            if isinstance(num_samples, (list, tuple, torch.Size)):
                shape = tuple(num_samples) + cov.shape
            else:
                shape = (num_samples,) + cov.shape
            cov = cov.expand(*shape)
        return cov

    def sample(self, mean, num_samples=None, **kwargs):
        shape = self.get_sample_shape(num_samples, mean.shape)
        return mean + torch.randn(*shape, device=mean.device) * self.std

    def log_prob(self, x, mean, num_samples=None, **kwargs):
        std = self.std
        var = std**2
        log_term = torch.log(2 * torch.pi * var)
        quad_term = ((x - mean) ** 2) / var
        return -0.5 * (log_term + quad_term).sum(dim=-1)

    def entropy(self, **kwargs):
        std = self.std
        var = std**2
        return 0.5 * torch.sum(
            torch.log(
                2 * torch.pi * torch.exp(torch.tensor(1.0, device=std.device)) * var
            )
        )
