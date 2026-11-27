import torch
import warnings
from model.Distributions.base_classes.base_distribution import BaseDistribution


class DistributionHandler:
    def __init__(self) -> None:
        # The respective posteriors/priors objects
        self.Distributions = dict()
        # If conditional: What is the input that has to be sampled first?
        self.SpecialInput = dict()
        # number of samples (only for posterior, really)
        self.num_samples = dict()
        # option to deactivate a posterior/prior (i.e. log_prob = 0, entropy = 0)
        self.active = dict()
        # for closed form updates
        self.ClosedFormUpdateAvailable = dict()
        self.closedFormUpdateActive = dict()
        # parameters on which you can call .expectation() instead of sample()
        self.expectation_active = dict()
        # list of all parameters (+ names) of all posteriors/priors
        self.parameters = []
        self.named_parameters = []

    def add(
        self,
        name: str,
        distribution: BaseDistribution,
        SpecialInput: list[str] | str | None = None,
        active: bool = True,
        closedFormUpdateAvailable: bool = False,
        expectation_active: bool = False,
    ) -> None:
        self.Distributions[name] = distribution
        self.SpecialInput[name] = SpecialInput
        self.active[name] = active
        self.ClosedFormUpdateAvailable[name] = closedFormUpdateAvailable
        if closedFormUpdateAvailable:
            self.closedFormUpdateActive[name] = True
        else:
            self.closedFormUpdateActive[name] = False
        self.expectation_active[name] = expectation_active
        # creates a list of all parameters of all posteriors/priors
        if distribution.parameters():
            self.parameters += list(distribution.parameters())
            self.named_parameters += list(distribution.named_parameters())
        else:
            warnings.warn(
                "No parameters found in posterior/prior/likelihood of name {}.".format(
                    name
                )
            )

    def sample(self, num_samples: int | dict[str, int]) -> dict[str, torch.Tensor]:
        """
        ASSUMPTIONs:
        - There are only first and zeroth order dependencies (not e.g. q(x|y) * q(y|z) * q(z))
        """
        samples = dict()
        # first sample without dependencies
        for name, distribution in self.Distributions.items():
            if self.SpecialInput[name] is None:
                if type(num_samples) is dict:
                    num_samples_val = num_samples[name]
                else:
                    num_samples_val = num_samples
                # check for dirty exception short cuts (closed form solution)
                if (
                    self.expectation_active[name]
                    and hasattr(distribution, "expectation")
                    and callable(distribution.expectation)
                ):
                    samples[name] = distribution.expectation()
                else:
                    samples[name] = distribution.sample(num_samples=num_samples_val)
        # then sample with dependencies
        for name, distribution in self.Distributions.items():
            if self.SpecialInput[name] is not None:
                if type(num_samples) is dict:
                    num_samples_val = num_samples[name]
                else:
                    num_samples_val = num_samples
                # check for dirty exception short cuts (closed form solution)
                if hasattr(distribution, "expectation") and callable(
                    distribution.expectation
                ):
                    samples[name] = distribution.expectation(
                        **self._handle_SpecialInput(name, samples)
                    )
                else:
                    samples[name] = distribution.sample(
                        num_samples=num_samples_val,
                        **self._handle_SpecialInput(name, samples),
                    )
        return samples

    def expected_log_prob(
        self, samples: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        log_prob = self.log_prob(samples)
        E_log_prob = dict()
        for name, distribution in self.Distributions.items():
            if hasattr(distribution, "expected_log_prob") and callable(
                distribution.expectation
            ):
                # check for dirty exception short cuts (closed form solution)
                E_log_prob[name] = distribution.expected_log_prob(
                    **self._handle_SpecialInput(name, samples)
                )
            else:
                # calculate the expectation by calculating the mean of the samples (Monte Carlo)
                E_log_prob[name] = torch.mean(log_prob[name], dim=0)
                # the expectation is a scalar, so if we are left with a 1D tensor, that means that we have multiple likelihoods, thus we need to sum it up
                if E_log_prob[name].squeeze().dim() == 1:
                    E_log_prob[name] = E_log_prob[name].sum()
            # deactivate if necessary
            if not self.active[name]:
                E_log_prob[name] = torch.zeros_like(E_log_prob[name])
        return E_log_prob, log_prob

    def log_prob(self, samples: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        log_prob = dict()
        for name, distribution in self.Distributions.items():
            # this sometimes also needs the own samples (if it is also in the posterior)
            if name in samples.keys():
                log_prob[name] = distribution.log_prob(
                    **{name: samples[name]}, **self._handle_SpecialInput(name, samples)
                )
            else:
                # or it doesnt, because it is basically a likelihood (not in the posterior and the values are calculated from other samples)
                log_prob[name] = distribution.log_prob(
                    **self._handle_SpecialInput(name, samples)
                )
            # deactivate if necessary
            if not self.active[name]:
                log_prob[name] = torch.zeros_like(log_prob[name])
        return log_prob

    def entropy(self, samples: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        entropy = dict()
        for name, distribution in self.Distributions.items():
            entropy[name] = distribution.entropy(
                **self._handle_SpecialInput(name, samples)
            )
            # deactivate if necessary
            if not self.active[name]:
                entropy[name] = torch.zeros_like(entropy[name])
        return entropy

    def expected_entropy(
        self, samples: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        entropy = self.entropy(samples)
        E_entropy = dict()
        for name, distribution in self.Distributions.items():
            if entropy[name].size() == torch.Size([]):
                E_entropy[name] = entropy[name]
            else:
                # calculate the expectation by calculating the mean of the samples (Monte Carlo)
                E_entropy[name] = torch.mean(entropy[name], dim=0)
                # the expectation is a scalar, so if we are left with a 1D tensor, that means that we have multiple likelihoods, thus we need to sum it up
                if E_entropy[name].squeeze().dim() == 1:
                    E_entropy[name] = E_entropy[name].sum()
            # deactivate if necessary
            if not self.active[name]:
                E_entropy[name] = torch.zeros_like(E_entropy[name])
        return E_entropy, entropy

    def closed_form_update(
        self, samples: dict[str, torch.Tensor], intermediates: dict[str, torch.Tensor]
    ) -> None:
        for name, distribution in self.Distributions.items():
            if self.active[name] and self.closedFormUpdateActive[name]:
                if callable(distribution.closedFormUpdate):
                    distribution.closedFormUpdate(samples, intermediates)

    def collect_intermediates(self) -> dict[str, torch.Tensor]:
        intermediates = dict()
        for name, distribution in self.Distributions.items():
            value, tag = distribution.collect_intermediates()
            if (value.numel() == 0) or (tag == ""):
                continue
            intermediates[tag] = value
        return intermediates

    def _handle_SpecialInput(
        self, name: str, samples: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Turns special input into a dictionary with the respective values from samples
        """
        SpecialInput = self.SpecialInput[name]
        if SpecialInput is None:
            return {}
        list_of_SpecialInput = list(SpecialInput)
        input_dict = {}
        for key in list_of_SpecialInput:
            input_dict[key] = samples[key]
        return input_dict

    def freeze_parameters(self, DistributionKey: str) -> None:
        self.Distributions[DistributionKey].freeze_parameters()
        self.closedFormUpdateActive[DistributionKey] = False

    def unfreeze_parameters(self, DistributionKey: str) -> None:
        self.Distributions[DistributionKey].unfreeze_parameters()
        if self.ClosedFormUpdateAvailable[DistributionKey]:
            self.closedFormUpdateActive[DistributionKey] = True

    def deactivate(self, DistributionKey: str) -> None:
        """
        Deactivates the posterior, i.e. sets the log_prob and entropy to 0.
        """
        self.active[DistributionKey] = False

    def activate(self, DistributionKey: str) -> None:
        """
        Activates the posterior, i.e. sets the log_prob and entropy to the original values.
        """
        self.active[DistributionKey] = True

    def state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        state_dict = dict()
        for name, distribution in self.Distributions.items():
            state_dict[name] = distribution.state_dict()
        return state_dict

    def load_state_dict(self, state_dict, strict=False, dont_load=[]) -> None:
        for name, distribution in self.Distributions.items():
            if name in dont_load:
                print("Not loading state_dict of posterior {}.".format(name))
                continue
            if name not in state_dict.keys():
                warnings.warn("No state_dict found for posterior {}.".format(name))
                continue
            distribution.load_state_dict(state_dict[name], strict=strict)
        print("Loaded successfully the state_dict of the posteriors.")

    def set_submodule_attrs(self, attr_dict: dict) -> None:
        """
        Set attributes of submodules in posteriors using the set_submodule_attr method.

        attr_dict: dict of the form
            {
                "posterior_name": {
                    "submodule_name": {
                        "attr_name": value,
                        ...
                    },
                    ...
                },
                ...
            }
        """
        for distribution_name, submods in attr_dict.items():
            if distribution_name not in self.Distributions:
                raise ValueError(f"Distribution '{distribution_name}' not found.")
            distribution = self.Distributions[distribution_name]
            for submodule_name, attrs in submods.items():
                for attr_name, value in attrs.items():
                    distribution.set_submodule_attr(submodule_name, attr_name, value)
