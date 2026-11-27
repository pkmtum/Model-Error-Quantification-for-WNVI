from model.Distributions.base_classes.base_distribution import BaseDistribution
from model.Distributions.normal_distributions.base_covariance_function import (
    CovarianceFunction,
)
from model.Distributions.mean_functions.base_mean_function import (
    MeanFunction,
)
from model.Distributions.observation_transformers.BaseObservationTransformer import (
    BaseObservationTransformer,
)
import torch


def _extract_num_samples(input_dict):
    # Pops num_samples from input_dict if present, else returns None
    if "num_samples" in input_dict:
        return input_dict.pop("num_samples")
    return None


class NormalDistribution(BaseDistribution):
    def __init__(
        self,
        prefix: str,
        mean_fn: MeanFunction,
        cov_fn: CovarianceFunction,
        observed_fn: BaseObservationTransformer | None = None,
    ) -> None:
        super().__init__(
            prefix=prefix,
            submodules={f"{prefix}_mean": mean_fn, f"{prefix}_covariance": cov_fn},
            observed_fn=observed_fn,
        )

    def mean_fn(self):
        return self.submodules[f"{self.prefix}_mean"]

    def cov_fn(self):
        return self.submodules[f"{self.prefix}_covariance"]

    def _filter_inputs(self, submodule, input_dict):
        return {k: v for k, v in input_dict.items() if k in submodule.special_inputs}

    def sample(self, **input_dict) -> torch.Tensor:
        """Sample from the distribution. Note that this method returns a sample of shape (num_samples, dim) in general case.
        Note that: If you are using a batch dimension for a input (e.g. in the neural network mean), it will return (batch_size, dim) and ignore num_samples.

        Returns:
            _type_: _description_
        """
        input_dict = dict(input_dict)  # avoid mutating caller's dict
        num_samples = _extract_num_samples(input_dict)
        mean = self.mean_fn().compute(**self._filter_inputs(self.mean_fn(), input_dict))
        return self.cov_fn().sample(
            mean,
            num_samples=num_samples,
            **self._filter_inputs(self.cov_fn(), input_dict),
        )

    def log_prob(self, **input_dict):
        input_dict = dict(input_dict)
        num_samples = _extract_num_samples(input_dict)
        # Use custom observed_fn if provided, else default to popping self.prefix
        if hasattr(self, "observed_fn") and self.observed_fn is not None:
            observed = self.observed_fn.apply(**input_dict)
            self._last_observed = (
                observed,
                getattr(self.observed_fn, "OutputName", ""),
            )
        else:
            observed = input_dict.pop(self.prefix)

        mean = self.mean_fn().compute(**self._filter_inputs(self.mean_fn(), input_dict))
        return self.cov_fn().log_prob(
            observed,
            mean,
            num_samples=num_samples,
            **self._filter_inputs(self.cov_fn(), input_dict),
        )

    def entropy(self, **kwargs):
        return self.cov_fn().entropy(**kwargs)
