import torch
from model.Distributions.base_classes.base_distribution import BaseDistribution
from model.Distributions.observation_transformers.BaseObservationTransformer import (
    BaseObservationTransformer,
)


def _extract_num_samples(input_dict):
    # Pops num_samples from input_dict if present, else returns None
    if "num_samples" in input_dict:
        return input_dict.pop("num_samples")
    return None


class BaseRateFunction(torch.nn.Module):
    """
    Base class for rate functions in exponential distributions.
    The rate parameter λ (lambda) is the inverse of the scale parameter.
    """

    def __init__(self, dim: int, special_inputs: list | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.special_inputs = special_inputs if special_inputs is not None else []

    def compute(self, num_samples=None, **kwargs) -> torch.Tensor:
        """
        Compute the rate parameter λ.

        Args:
            num_samples: Number of samples (may be ignored by some implementations)
            **kwargs: Additional inputs required by the rate function

        Returns:
            torch.Tensor: Rate parameter λ with shape (..., dim)
        """
        raise NotImplementedError


class ExponentialDistribution(BaseDistribution):
    """
    Exponential distribution with configurable rate function.

    The exponential distribution has PDF: f(x) = λ * exp(-λ * x) for x ≥ 0
    where λ > 0 is the rate parameter.
    """

    def __init__(
        self,
        prefix: str,
        rate_fn: BaseRateFunction,
        observed_fn: BaseObservationTransformer | None = None,
    ) -> None:
        super().__init__(
            prefix=prefix,
            submodules={f"{prefix}_rate": rate_fn},
            observed_fn=observed_fn,
        )

    def rate_fn(self):
        return self.submodules[f"{self.prefix}_rate"]

    def _filter_inputs(self, submodule, input_dict):
        return {k: v for k, v in input_dict.items() if k in submodule.special_inputs}

    def sample(self, **input_dict) -> torch.Tensor:
        """
        Sample from the exponential distribution.

        Returns:
            torch.Tensor: Samples with shape (num_samples, dim) or (batch_size, dim)
        """
        input_dict = dict(input_dict)  # avoid mutating caller's dict
        num_samples = _extract_num_samples(input_dict)
        rate_inputs = self._filter_inputs(self.rate_fn(), input_dict)
        rate = self.rate_fn().compute(num_samples=num_samples, **rate_inputs)

        # Sample from exponential: X ~ Exp(λ) can be generated as -ln(U)/λ where U ~ Uniform(0,1)
        if num_samples is not None:
            if isinstance(num_samples, (list, tuple, torch.Size)):
                shape = tuple(num_samples) + rate.shape
            else:
                shape = (num_samples,) + rate.shape
        else:
            shape = rate.shape

        uniform_samples = torch.rand(*shape, device=rate.device)
        return -torch.log(uniform_samples) / rate

    def log_prob(self, **input_dict) -> torch.Tensor:
        """
        Compute log probability of x under the exponential distribution.

        Args:
            **input_dict: Inputs for the rate function and observations

        Returns:
            torch.Tensor: Log probabilities with shape (...)
        """
        input_dict = dict(input_dict)  # avoid mutating caller's dict
        num_samples = input_dict.pop("num_samples", None)

        # Use custom observed_fn if provided, else default to popping self.prefix
        if hasattr(self, "observed_fn") and self.observed_fn is not None:
            x = self.observed_fn.apply(**input_dict)
            self._last_observed = (
                x,
                getattr(self.observed_fn, "OutputName", ""),
            )
        else:
            x = input_dict.pop(self.prefix)

        rate_inputs = self._filter_inputs(self.rate_fn(), input_dict)
        rate = self.rate_fn().compute(num_samples=num_samples, **rate_inputs)

        # Log PDF: log(λ) - λ * x for x ≥ 0
        # We assume x ≥ 0 (should be enforced by user)
        log_prob = torch.log(rate) - rate * x
        return log_prob.sum(dim=-1)  # Sum over the last dimension (features)

    def entropy(self, **input_dict) -> torch.Tensor:
        """
        Compute entropy of the exponential distribution.

        Returns:
            torch.Tensor: Entropy values
        """
        input_dict = dict(input_dict)  # avoid mutating caller's dict
        num_samples = _extract_num_samples(input_dict)
        rate_inputs = self._filter_inputs(self.rate_fn(), input_dict)
        rate = self.rate_fn().compute(num_samples=num_samples, **rate_inputs)

        # Entropy: 1 - log(λ)
        entropy = 1.0 - torch.log(rate)
        return entropy.sum(dim=-1)  # Sum over the last dimension (features)

    def mean(self, **input_dict) -> torch.Tensor:
        """
        Compute mean of the exponential distribution.

        Returns:
            torch.Tensor: Mean values (1/λ)
        """
        input_dict = dict(input_dict)  # avoid mutating caller's dict
        num_samples = _extract_num_samples(input_dict)
        rate_inputs = self._filter_inputs(self.rate_fn(), input_dict)
        rate = self.rate_fn().compute(num_samples=num_samples, **rate_inputs)
        return 1.0 / rate

    def variance(self, **input_dict) -> torch.Tensor:
        """
        Compute variance of the exponential distribution.

        Returns:
            torch.Tensor: Variance values (1/λ²)
        """
        input_dict = dict(input_dict)  # avoid mutating caller's dict
        num_samples = _extract_num_samples(input_dict)
        rate_inputs = self._filter_inputs(self.rate_fn(), input_dict)
        rate = self.rate_fn().compute(num_samples=num_samples, **rate_inputs)
        return 1.0 / (rate**2)
