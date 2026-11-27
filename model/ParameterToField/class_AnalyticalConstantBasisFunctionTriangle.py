import torch
from model.ParentClasses.class_ParameterToField import ParameterToField


class AnalyticalConstantBasisFunction(ParameterToField):
    def __init__(self, options: dict) -> None:
        super().__init__("center_points", 1, options["bf_args"], options["bf_grid"])

    def eval(
        self, parameter: torch.Tensor, apply_known_values: bool = True
    ) -> torch.Tensor:
        if apply_known_values:
            parameter = self.apply_know_values(parameter)
        return parameter

    def eval_grad_s(
        self, parameter: torch.Tensor, apply_known_values: bool = True
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Gradients are not available for AnalyticalConstantBasisFunction, as the field consists of piecewise constant elements."
        )
