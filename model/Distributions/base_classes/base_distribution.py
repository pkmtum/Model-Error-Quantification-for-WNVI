import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, Mapping
from model.Distributions.observation_transformers.BaseObservationTransformer import (
    BaseObservationTransformer,
)


class BaseDistribution(nn.Module):
    """Base class for all distributions.

    Args:
        nn (Module): PyTorch neural network module.
    """

    def __init__(
        self,
        prefix: str,
        submodules: Mapping[str, nn.Module],
        observed_fn: BaseObservationTransformer | None = None,
    ):
        """Initialize the base distribution.

        Args:
            prefix (str): Prefix for the distribution. Determines the naming of parameters.
            submodules (Mapping[str, nn.Module]): Submodules of the distribution. Typically with naming convention "{prefix}_submodule_name".
            observed_fn (BaseObservationTransformer | None, optional): Function to transform variables into whatever should be handled within the distribution (e.g. residuals, jumps between elements). Defaults to None.
        """
        super().__init__()
        self.prefix = prefix
        self.submodules = nn.ModuleDict(submodules)
        self.observed_fn = observed_fn
        self._last_observed = (torch.tensor([]), "")

    def _filter_inputs(
        self, submodule: nn.Module, input_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        if hasattr(submodule, "special_inputs"):
            return {
                k: v for k, v in input_dict.items() if k in submodule.special_inputs
            }
        return input_dict

    def freeze_parameters(self, param_names: Optional[list[str]] = None) -> None:
        if param_names is None:
            for _, param in self.named_parameters():
                param.requires_grad = False
        else:
            for name, param in self.named_parameters():
                param.requires_grad = name in param_names

    def unfreeze_parameters(self, param_names: Optional[list[str]] = None) -> None:
        if param_names is None:
            for _, param in self.named_parameters():
                param.requires_grad = True
        else:
            for name, param in self.named_parameters():
                param.requires_grad = name in param_names

    def set_parameter(self, parameter_name: str, value: torch.Tensor) -> None:
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name == parameter_name:
                    param.copy_(value)
                    return
        raise ValueError(f"Parameter '{parameter_name}' not found.")

    def collect_intermediates(self) -> Tuple[torch.Tensor, str]:
        return self._last_observed

    def set_submodule_attr(
        self, submodule_name: str, attr_name: str, value: Any
    ) -> None:
        """
        Set an attribute of a submodule (not limited to nn.Parameter).
        """
        if submodule_name not in self.submodules:
            raise ValueError(f"Submodule '{submodule_name}' not found.")

        if not hasattr(self.submodules[submodule_name], attr_name):
            raise ValueError(
                f"Attribute '{attr_name}' not found in submodule '{submodule_name}'."
            )

        if isinstance(
            getattr(self.submodules[submodule_name], attr_name), torch.Tensor
        ):
            value = value.to(getattr(self.submodules[submodule_name], attr_name).device)

        setattr(self.submodules[submodule_name], attr_name, value)

        print(
            f"Set attribute '{attr_name}' of submodule '{submodule_name}' to {value}."
        )
