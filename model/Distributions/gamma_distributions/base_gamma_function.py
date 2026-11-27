from model.Distributions.base_classes.base_distribution import BaseDistribution


class ModularGammaDistribution(BaseDistribution):
    def __init__(self, prefix, gamma_fn, observed_fn=None):
        submodules = {f"{prefix}_gamma": gamma_fn}
        super().__init__(prefix, submodules, observed_fn)

    def gamma_fn(self):
        return self.submodules[f"{self.prefix}_gamma"]

    def sample(self, **input_dict):
        input_dict = dict(input_dict)
        num_samples = input_dict.pop("num_samples", None)
        return self.gamma_fn().sample(
            num_samples=num_samples, **self._filter_inputs(self.gamma_fn(), input_dict)
        )

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
        return self.gamma_fn().log_prob(
            observed, **self._filter_inputs(self.gamma_fn(), input_dict)
        )

    def entropy(self, **input_dict):
        return self.gamma_fn().entropy(
            **self._filter_inputs(self.gamma_fn(), input_dict)
        )

    def mean(self, **input_dict):
        return self.gamma_fn().mean(**self._filter_inputs(self.gamma_fn(), input_dict))

    def expectation(self, **input_dict):
        return self.mean(**input_dict)

    def closedFormUpdate(self, samples, intermediates):
        gamma_fn = self.gamma_fn()
        if hasattr(gamma_fn, "closedFormUpdate"):
            return gamma_fn.closedFormUpdate(samples, intermediates)
        raise NotImplementedError(
            "closedFormUpdate is not implemented for this gamma function."
        )
