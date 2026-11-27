import torch
from model.Distributions.base_classes.base_distribution import BaseDistribution


class DeltaDistribution(BaseDistribution):
    def __init__(self, prefix, mean_fn, observed_fn=None):
        submodules = {f"{prefix}_mean": mean_fn}
        super().__init__(prefix, submodules, observed_fn)
        self.zero = torch.tensor(0.0)

    def mean_fn(self):
        return self.submodules[f"{self.prefix}_mean"]

    def sample(self, **input_dict):
        input_dict = dict(input_dict)
        mean = self.mean_fn().compute(**self._filter_inputs(self.mean_fn(), input_dict))
        return mean

    def log_prob(self, **input_dict):
        input_dict = dict(input_dict)
        if self.observed_fn is not None:
            observed = self.observed_fn.apply(**input_dict)
            self._last_observed = (
                observed,
                getattr(self.observed_fn, "OutputName", ""),
            )
        else:
            observed = input_dict.pop(self.prefix)
        # Dirac delta: log_prob is zero everywhere except at mean, but for practical purposes return zeros
        return self.zero.expand(observed.shape[0])

    def entropy(self, **kwargs):
        return self.zero
