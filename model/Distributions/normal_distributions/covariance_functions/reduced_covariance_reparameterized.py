import torch
import torch.nn as nn
from model.Distributions.normal_distributions.base_covariance_function import (
    CovarianceFunction,
)
from utils.torch_funcs.function_make_torch_tensor import make_torch_tensor


class ReducedCovarianceReparameterized(CovarianceFunction):
    """
    Reduced-rank covariance with reparameterized V matrix for more stable training.

    Instead of directly learning V, we parameterize each element as:
    V[i,j] = sign[i,j] * magnitude[i,j] = tanh(sign_logit[i,j]) * exp(log_magnitude[i,j])

    This separates the magnitude and sign, leading to more stable gradient flow.
    """

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

        # Initialize reparameterized V
        # If V is provided as initialization, decompose it into magnitude and sign
        init_V = torch.ones(dim, reduced_dim) * V

        # Initialize log_magnitude and sign_logit
        # For small initial V values, we use small random initialization
        if torch.allclose(init_V, torch.zeros_like(init_V)):
            # Small random initialization
            self.log_magnitude = nn.Parameter(
                torch.randn(dim, reduced_dim) * 0.01 - 3.0
            )
            self.sign_logit = nn.Parameter(torch.randn(dim, reduced_dim) * 0.1)
        else:
            # Initialize from provided V
            magnitude = torch.abs(init_V) + 1e-8  # avoid log(0)
            sign = torch.sign(init_V)
            # Convert sign from {-1, 0, 1} to sign_logit space via inverse tanh
            # tanh(x) ≈ sign, so x ≈ atanh(sign)
            # For sign=0, we use 0; for ±1, we use a large value
            sign_logit_init = torch.where(
                torch.abs(sign) > 0.5,
                sign * 3.0,  # Large value to give tanh ≈ ±1
                torch.zeros_like(sign),
            )
            self.log_magnitude = nn.Parameter(torch.log(magnitude))
            self.sign_logit = nn.Parameter(sign_logit_init)

        self.log_sigma = nn.Parameter(init_sigma_log.clone())

    @property
    def V(self) -> torch.Tensor:
        """Reconstruct V from reparameterized form."""
        magnitude = torch.exp(self.log_magnitude)
        sign = torch.tanh(self.sign_logit)
        return sign * magnitude

    def compute(self, num_samples=None, **kwargs):
        sigma2 = torch.exp(self.log_sigma) ** 2
        V = self.V  # Use the property to get reconstructed V
        cov = torch.diag_embed(sigma2) + V @ V.t()
        if num_samples is not None:
            if isinstance(num_samples, (list, tuple, torch.Size)):
                shape = tuple(num_samples) + cov.shape
            else:
                shape = (num_samples,) + cov.shape
            cov = cov.expand(*shape)
        return cov

    def sample(self, mean, num_samples=None, **kwargs):
        sigma = torch.exp(self.log_sigma)
        V = self.V  # Use the property to get reconstructed V

        # This is going to be the final shape of the sample
        shape = self.get_sample_shape(num_samples, mean.shape)
        diag_noise = torch.randn(*shape, device=mean.device) * sigma
        low_rank_noise = (
            torch.randn(*shape[:-1], self.reduced_dim, device=mean.device) @ V.T
        )
        return mean + diag_noise + low_rank_noise

    def log_prob(self, x, mean, num_samples=None, **kwargs):
        raise NotImplementedError(
            "Log prob not implemented for ReducedCovarianceReparameterized."
        )

    def entropy(self, **kwargs):
        sigma2 = torch.exp(self.log_sigma) ** 2
        V = self.V  # Use the property to get reconstructed V

        logdet_A = torch.sum(torch.log(sigma2))
        Ainv_V = V.t() * (1 / sigma2).unsqueeze(0)
        Vt_Ainv_V = Ainv_V @ V
        logdet_update = torch.logdet(
            Vt_Ainv_V + torch.eye(self.reduced_dim, device=V.device)
        )
        return 0.5 * (logdet_A + logdet_update)
