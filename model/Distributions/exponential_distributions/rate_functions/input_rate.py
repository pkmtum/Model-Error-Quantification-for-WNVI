import torch
from model.Distributions.exponential_distributions.base_exponential_distribution import (
    BaseRateFunction,
)


class InputRate(BaseRateFunction):
    """
    Input-dependent rate parameter for exponential distribution.
    The rate λ is provided as input, similar to InputDiagonalPrecision in normal distributions.
    This allows the rate to depend on other variables in the model.
    """

    def __init__(self, dim: int, rate_input_key: str = "lmbda") -> None:
        """
        Initialize input-dependent rate function.

        Args:
            dim: Dimension of the exponential distribution
            rate_input_key: Key name for the rate input in kwargs
        """
        super().__init__(dim=dim, special_inputs=[rate_input_key])
        self.rate_input_key = rate_input_key

    def compute(self, num_samples=None, **kwargs) -> torch.Tensor:
        """
        Compute the rate parameter from input.

        Args:
            num_samples: Number of samples (may affect shape expansion)
            **kwargs: Must contain the rate input specified by rate_input_key

        Returns:
            torch.Tensor: Rate parameter with appropriate shape
        """
        if self.rate_input_key not in kwargs:
            raise KeyError(f"Rate input '{self.rate_input_key}' not found in inputs")

        rate = kwargs[self.rate_input_key]

        # Ensure rate is positive
        if torch.any(rate <= 0):
            raise ValueError("All rate parameters must be positive")

        # Expand to appropriate shape if needed
        if num_samples is not None:
            if isinstance(num_samples, (list, tuple, torch.Size)):
                shape = tuple(num_samples) + rate.shape
            else:
                shape = (num_samples,) + rate.shape
            rate = rate.expand(*shape)

        return rate
