import torch
from model.class_ELBO import ELBO
from typing import Callable


class SVI:
    def __init__(
        self,
        elbo: ELBO,
        num_posterior_samples: int = 1,
        default_scheduler_init_fn: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
        ]
        | None = None,
        default_learning_rate: float = 1e-3,
    ) -> None:
        self.elbo = elbo
        self.num_posterior_samples = num_posterior_samples
        self.default_learning_rate = default_learning_rate
        self.default_scheduler_init_fn = default_scheduler_init_fn
        self.optimizer = torch.optim.Adam(
            self.elbo.posteriors.parameters,
            lr=self.default_learning_rate,
            maximize=True,
        )
        self.scheduler = (
            self.default_scheduler_init_fn(self.optimizer)
            if self.default_scheduler_init_fn
            else None
        )

    def reset(
        self,
        learning_rate: float | None = None,
        scheduler_init_fn: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
        ]
        | None = None,
    ) -> None:
        self.optimizer = torch.optim.Adam(
            self.elbo.posteriors.parameters,
            lr=learning_rate
            if learning_rate is not None
            else self.default_learning_rate,
            maximize=True,
        )
        if scheduler_init_fn is not None:
            self.scheduler = scheduler_init_fn(self.optimizer)
        else:
            self.scheduler = (
                self.default_scheduler_init_fn(self.optimizer)
                if self.default_scheduler_init_fn
                else None
            )

    def step(self):
        # reset the optimizer
        self.optimizer.zero_grad()

        # Get Evidence Lower Bound (ELBO) and its components
        samples = self.elbo.sample(num_samples=self.num_posterior_samples)
        elbo, E_log_likelihoods, E_log_priors, E_entropy_posteriors = self.elbo.eval(
            samples
        )

        # compute gradients of the parameters with respect to the ELBO
        elbo.backward(retain_graph=True)

        # do a step of the optimizer
        self.optimizer.step()

        # do a step
        if self.scheduler is not None:
            self.scheduler.step()

        # perform closed form updates of parameters
        intermediates = self.elbo.collect_intermediates()
        self.elbo.closed_form_update(samples, intermediates)

        # get the current gradient
        current_grad = dict()
        for name, param in self.elbo.posteriors.named_parameters:
            if (
                param.requires_grad
            ):  # there may be frozen parameters that do not give a gradient
                current_grad[name] = param.grad.clone().detach()

        return (
            elbo,
            current_grad,
            samples,
            intermediates,
            E_log_likelihoods,
            E_log_priors,
            E_entropy_posteriors,
        )


# # prepare quantities for logging
# samples_detached = dict()
# samples_detached = collect_dict_entries(samples_detached, samples, "")


# def prepare_elbo_parts_for_logging(list_of_elbo_parts: list[dict[str, torch.Tensor]],
#                                    name_prefixes: list[str] = [""]
#                                    ) -> dict[str, torch.Tensor]:
#     """Prepare ELBO parts for logging by flattening the list of dictionaries into a single dictionary.

#     Args:
#         list_of_elbo_parts (list[dict[str, torch.Tensor]]): A list of dictionaries containing ELBO parts.
#         name_prefixes (list[str], optional): A list of prefixes to prepend to the keys in the ELBO parts dictionaries. Defaults to [""].

#     Returns:
#         dict[str, torch.Tensor]: A single dictionary containing all ELBO parts with their keys {prefix}+{name}.
#     """
#     elbo_parts = dict()
#     if name_prefixes == [""]:
#         name_prefixes = [""] * len(list_of_elbo_parts)
#     else:
#         assert len(name_prefixes) == len(list_of_elbo_parts), "Number of name prefixes must match the number of ELBO parts dictionaries."

#     for elbo_part, name_prefix in zip(list_of_elbo_parts, name_prefixes):
#         elbo_parts = collect_dict_entries(elbo_parts, elbo_part, name_prefix)

#     return elbo_parts


# def collect_dict_entries(target_dict: dict[str, torch.Tensor],
#                        sample_dict: dict[str, torch.Tensor],
#                        key_prefix: str,
#                        ) -> dict[str, torch.Tensor]:
#     """Collect ELBO parts, clone and detach them to avoid in-place operations.

#     Args:
#         target_dict (dict[str, torch.Tensor]): Dictionary of ELBO parts.
#         sample_dict (dict[str, torch.Tensor]): Dictionary of ELBO entries.

#     Returns:
#         tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: A tuple containing the collected ELBO parts and samples.
#     """
#     for name, sample in sample_dict.items():
#         sample_dict[key_prefix + name] = sample.clone().detach()
#     return target_dict
