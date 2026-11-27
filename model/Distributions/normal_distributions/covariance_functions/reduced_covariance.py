import torch
import torch.nn as nn
from model.Distributions.normal_distributions.base_covariance_function import (
    CovarianceFunction,
)
from utils.torch_funcs.function_make_torch_tensor import make_torch_tensor


class ReducedCovariance(CovarianceFunction):
    def __init__(
        self,
        dim: int,
        reduced_dim: int,
        std: float | torch.Tensor = 1.0,
        V: float | torch.Tensor = 0.0,
    ) -> None:
        super().__init__(dim=dim, std=std)
        self.special_inputs = []
        self.reduced_dim = reduced_dim

        init_sigma_log = make_torch_tensor(std, torch.Size([dim])).log()
        init_V = torch.ones(dim, reduced_dim) * V

        self.log_sigma = nn.Parameter(init_sigma_log.clone())
        self.V = nn.Parameter(init_V.clone())

    def compute(self, num_samples=None, **kwargs):
        sigma2 = torch.exp(self.log_sigma) ** 2
        cov = torch.diag_embed(sigma2) + self.V @ self.V.t()
        if num_samples is not None:
            if isinstance(num_samples, (list, tuple, torch.Size)):
                shape = tuple(num_samples) + cov.shape
            else:
                shape = (num_samples,) + cov.shape
            cov = cov.expand(*shape)
        return cov

    def sample(self, mean, num_samples=None, **kwargs):
        sigma = torch.exp(self.log_sigma)

        # This is going to be the final shape of the sample
        shape = self.get_sample_shape(num_samples, mean.shape)
        diag_noise = torch.randn(*shape, device=mean.device) * sigma
        low_rank_noise = (
            torch.randn(*shape[:-1], self.reduced_dim, device=mean.device) @ self.V.T
        )
        return mean + diag_noise + low_rank_noise

    def log_prob(self, x, mean, num_samples=None, **kwargs):
        raise NotImplementedError("Log prob not implemented for ReducedCovariance.")

    def entropy(self, **kwargs):
        sigma2 = torch.exp(self.log_sigma) ** 2
        logdet_A = torch.sum(torch.log(sigma2))
        Ainv_V = self.V.t() * (1 / sigma2).unsqueeze(0)
        Vt_Ainv_V = Ainv_V @ self.V
        logdet_update = torch.logdet(
            Vt_Ainv_V + torch.eye(self.reduced_dim, device=self.V.device)
        )
        return 0.5 * (logdet_A + logdet_update)
