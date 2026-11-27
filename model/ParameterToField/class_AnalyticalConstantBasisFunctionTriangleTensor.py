import torch
from model.ParentClasses.class_ParameterToField import ParameterToField


class AnalyticalConstantBasisFunctionTensor(ParameterToField):
    def __init__(self, options: dict) -> None:
        super().__init__("center_points", 3, options["bf_args"], options["bf_grid"])

    def eval(
        self, parameter: torch.Tensor, apply_known_values: bool = True
    ) -> torch.Tensor:
        if apply_known_values:
            parameter = self.apply_know_values(parameter)
        xi_xx = (
            parameter[..., : self.n_bfs_per_field].unsqueeze(-1).unsqueeze(-1)
        )  # create a 2x2 tensor
        xi_xy = (
            parameter[..., self.n_bfs_per_field : (2 * self.n_bfs_per_field)]
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        xi_yy = parameter[..., (2 * self.n_bfs_per_field) :].unsqueeze(-1).unsqueeze(-1)
        sigma = torch.cat(
            [torch.cat([xi_xx, xi_xy], dim=-1), torch.cat([xi_xy, xi_yy], dim=-1)],
            dim=-2,
        )
        return sigma  # [... x Nele x 2 x 2]

    def eval_grad_s(
        self, parameter: torch.Tensor, apply_known_values: bool = True
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Gradients are not available for AnalyticalConstantBasisFunctionTensor, as the field consists of piecewise constant elements."
        )
