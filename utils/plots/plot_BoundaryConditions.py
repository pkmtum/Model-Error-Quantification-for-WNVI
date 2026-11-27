import torch
import matplotlib.tri
from model.ParameterToField.class_AnalyticalLinearBasisFunctionTriangle import (
    AnalyticalLinearBasisFunction,
)
from utils.plots.plot_multiple_fields import plot_multiple_fields


def plot_Neumann_BCs(
    RightHandSideOptions: dict,
    ThetaToField: AnalyticalLinearBasisFunction,
    triangulation: matplotlib.tri.Triangulation,
    result_path: str,
) -> None:
    Neumann_theta = torch.zeros(int(ThetaToField.num_unknowns))
    mask = getattr(
        ThetaToField,
        "mask_given_values",
        torch.ones(int(ThetaToField.num_unknowns), dtype=torch.bool),
    )
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.bool)
    mask = mask.bool()

    if "neumann_left" in RightHandSideOptions:
        Theta1 = ThetaToField.left_node_list * RightHandSideOptions["neumann_left"][0]
        Theta2 = ThetaToField.left_node_list * RightHandSideOptions["neumann_left"][1]
        a = torch.concat((Theta1, Theta2), dim=0)
        Neumann_theta += a[~mask]
    if "neumann_right" in RightHandSideOptions:
        Theta1 = ThetaToField.right_node_list * RightHandSideOptions["neumann_right"][0]
        Theta2 = ThetaToField.right_node_list * RightHandSideOptions["neumann_right"][1]
        a = torch.concat((Theta1, Theta2), dim=0)
        Neumann_theta += a[~mask]
    if "neumann_top" in RightHandSideOptions:
        Theta1 = ThetaToField.top_node_list * RightHandSideOptions["neumann_top"][0]
        Theta2 = ThetaToField.top_node_list * RightHandSideOptions["neumann_top"][1]
        a = torch.concat((Theta1, Theta2), dim=0)
        Neumann_theta += a[~mask]
    if "neumann_bottom" in RightHandSideOptions:
        Theta1 = (
            ThetaToField.bottom_node_list * RightHandSideOptions["neumann_bottom"][0]
        )
        Theta2 = (
            ThetaToField.bottom_node_list * RightHandSideOptions["neumann_bottom"][1]
        )
        a = torch.concat((Theta1, Theta2), dim=0)
        Neumann_theta += a[~mask]

    Neumann_plot1 = ThetaToField.apply_know_values(Neumann_theta)
    Neumann_plot = (
        torch.unflatten(
            Neumann_plot1,
            -1,
            (
                2,
                ThetaToField.bf_grid.size(dim=-2),
                ThetaToField.bf_grid.size(dim=-1),
            ),
        )
        .flatten(start_dim=-2)
        .transpose(dim0=-2, dim1=-1)
    )

    plot_multiple_fields(
        Neumann_plot.squeeze(),
        triangulation,
        titles=[
            r"$Neumann BCS_{theta1}$",
            r"$Neumann BCS_{theta2}$",
        ],
        results_path=result_path,
    )


def plot_Dirichlet_BCs(
    ThetaToField: AnalyticalLinearBasisFunction,
    triangulation: matplotlib.tri.Triangulation,
    result_path: str,
) -> None:
    Dirichlet_plot1 = ThetaToField.apply_know_values(
        torch.ones(int(ThetaToField.num_unknowns))
    )
    Dirichlet_plot = (
        torch.unflatten(
            Dirichlet_plot1,
            -1,
            (
                2,
                ThetaToField.bf_grid.size(dim=-2),
                ThetaToField.bf_grid.size(dim=-1),
            ),
        )
        .flatten(start_dim=-2)
        .transpose(dim0=-2, dim1=-1)
    )

    plot_multiple_fields(
        Dirichlet_plot.squeeze(),
        triangulation,
        titles=[
            r"$Dirichlet BCS_{theta1}$",
            r"$Dirichlet BCS_{theta2}$",
        ],
        results_path=result_path,
    )
