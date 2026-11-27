import torch
from model.Distributions.exponential_distributions.base_exponential_distribution import (
    BaseRateFunction,
)


class LearnableVectorRate(BaseRateFunction):
    """
    Learnable vector rate parameter for exponential distribution.
    The rate λ is a learnable parameter vector.
    """

    def __init__(self, dim: int, initial_lmbda: float = 1.0) -> None:
        """
        Initialize learnable vector rate function.

        Args:
            dim: Dimension of the exponential distribution
            initial_lmbda: Initial value for the rate parameter (λ > 0)
        """
        super().__init__(dim=dim, special_inputs=[])
        if initial_lmbda <= 0:
            raise ValueError("Initial rate parameter lambda must be positive")

        # Use log parameterization to ensure positivity: λ = exp(log_lmbda)
        log_initial_lmbda = torch.log(torch.tensor(initial_lmbda))
        self.log_lmbda = torch.nn.Parameter(log_initial_lmbda.expand(dim).clone())

    def compute(self, num_samples=None, **kwargs) -> torch.Tensor:
        """
        Compute the learnable rate parameter vector.

        Args:
            num_samples: Number of samples (ignored for learnable rate)
            **kwargs: Additional inputs (ignored)

        Returns:
            torch.Tensor: Rate parameter with shape (dim,)
        """
        return torch.exp(self.log_lmbda)

    @property
    def lmbda(self) -> torch.Tensor:
        """Get the current rate parameter values."""
        return torch.exp(self.log_lmbda)
