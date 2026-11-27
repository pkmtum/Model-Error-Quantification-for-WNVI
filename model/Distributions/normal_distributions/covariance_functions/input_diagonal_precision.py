import torch
from model.Distributions.normal_distributions.base_covariance_function import (
    CovarianceFunction,
)


class InputDiagonalPrecision(CovarianceFunction):
    """
    Diagonal covariance where the precision (inverse variance) is provided as input,
    not as a learnable parameter or buffer.
    The input should be a vector of precisions (1/variance) for a diagonal covariance structure.
    The name of the precision input can be set via precision_input_key.
    """

    def __init__(
        self, dim: int, precision_input_key: str = "precision", std=0.0
    ) -> None:
        super().__init__(dim=dim, special_inputs=[precision_input_key], std=std)
        self.precision_input_key = precision_input_key

    def compute(self, num_samples=None, **kwargs):
        precision = kwargs[self.precision_input_key]
        var = 1.0 / precision
        cov = torch.diag_embed(var)
        if num_samples is not None:
            if isinstance(num_samples, (list, tuple, torch.Size)):
                shape = tuple(num_samples) + cov.shape
            else:
                shape = (num_samples,) + cov.shape
            cov = cov.expand(*shape)
        return cov

    def sample(self, mean, num_samples=None, **kwargs):
        precision = kwargs[self.precision_input_key]
        std = torch.sqrt(1.0 / precision)
        shape = self.get_sample_shape(num_samples, mean.shape)
        return mean + torch.randn(*shape, device=mean.device) * std

    def log_prob(self, x, mean, num_samples=None, **kwargs):
        precision = kwargs[self.precision_input_key]
        var = 1.0 / precision
        log_term = torch.log(2 * torch.pi * var)
        quad_term = ((x - mean) ** 2) * precision
        return -0.5 * (log_term + quad_term).sum(dim=-1)

    def entropy(self, **kwargs):
        precision = kwargs[self.precision_input_key]
        var = 1.0 / precision
        # H = 0.5 * sum(log(2 * pi * e * var))
        return 0.5 * torch.sum(
            torch.log(
                2
                * torch.pi
                * torch.exp(torch.tensor(1.0, device=precision.device))
                * var
            )
        )
