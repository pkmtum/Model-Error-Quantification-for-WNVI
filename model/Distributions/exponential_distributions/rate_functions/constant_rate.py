import torch
from model.Distributions.exponential_distributions.base_exponential_distribution import (
    BaseRateFunction,
)


class ConstantRate(BaseRateFunction):
    """
    Constant rate parameter for exponential distribution.
    The rate λ is a fixed constant value.
    """

    def __init__(self, dim: int, lmbda: float = 1.0) -> None:
        """
        Initialize constant rate function.

        Args:
            dim: Dimension of the exponential distribution
            lmbda: Constant rate parameter (λ > 0)
        """
        super().__init__(dim=dim, special_inputs=[])
        if lmbda <= 0:
            raise ValueError("Rate parameter lambda must be positive")
        self.register_buffer("lmbda", torch.tensor(lmbda))

    def compute(self, num_samples=None, **kwargs) -> torch.Tensor:
        """
        Compute the constant rate parameter.

        Args:
            num_samples: Number of samples (ignored for constant rate)
            **kwargs: Additional inputs (ignored)

        Returns:
            torch.Tensor: Rate parameter with shape (dim,)
        """
        return self.lmbda.expand(self.dim)
