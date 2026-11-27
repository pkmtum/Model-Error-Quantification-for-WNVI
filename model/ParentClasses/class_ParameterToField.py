import torch
from abc import ABC, abstractmethod
from utils.Integrator.function_integrator_2D import trapzInt2D
from utils.torch_funcs.function_LinearTriangle import get_triangular_nodes


class ParameterToField(ABC):
    def __init__(
        self,
        n_bfs_option: str,
        field_dimension: int,
        bf_args: dict,
        bf_grid: torch.Tensor,
    ) -> None:
        self.bf_args = bf_args
        self.bf_grid = bf_grid
        self.nodes, self.nodesMap = get_triangular_nodes(self.bf_grid)
        self.center_points = torch.mean(torch.stack(self.nodes)[:, :2, :], dim=2).T

        if n_bfs_option == "grid":
            self.n_bfs_per_field = self.bf_grid[0].numel()
            self.n_bfs = self.n_bfs_per_field * field_dimension
        elif n_bfs_option == "center_points":
            self.n_bfs = self.center_points.shape[1] * field_dimension
            self.n_bfs_per_field = self.center_points.shape[1]
        else:
            raise ValueError("n_bfs_option must be 'nodes' or 'center_points'")

        self.flag_given_values = bf_args.get("flag_given_values", False)
        if self.flag_given_values:
            self.given_value = bf_args["given_values"]
            self.mask_given_values = bf_args["mask_given_values"].flatten()
            self.num_given_values = torch.sum(self.mask_given_values)
            self.num_unknowns = self.n_bfs - self.num_given_values

            # Precalculated values for faster application
            # Tensor with boundary condition values (n_bfs,)
            applied_given_values = torch.zeros(self.n_bfs)
            applied_given_values[self.mask_given_values] = self.given_value
            self.applied_given_values = applied_given_values

            # Tensor to map parameters (..., num_unknowns) to (..., n_bfs)
            unknown_mask = (~self.mask_given_values).float()  # shape (100,)
            unknown_indices = unknown_mask.nonzero(as_tuple=True)[0]  # shape (88,)
            ParameterToNbfsMap = torch.zeros(int(self.num_unknowns), self.n_bfs)
            ParameterToNbfsMap[
                torch.arange(int(self.num_unknowns)), unknown_indices
            ] = 1.0
            self.ParameterToNbfsMap = ParameterToNbfsMap
        else:
            self.num_unknowns = torch.tensor(self.n_bfs)

        self.flag_norm = bf_args.get("flag_norm", True)

    @abstractmethod
    def eval(
        self, parameter: torch.Tensor, apply_known_values: bool = True
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def eval_grad_s(
        self, parameter: torch.Tensor, apply_known_values: bool = True
    ) -> torch.Tensor:
        pass

    def eval_WF(self, parameter: torch.Tensor) -> torch.Tensor:
        WF = self.eval(parameter)
        if not self.flag_norm:
            return WF
        norm_WF = trapzInt2D(torch.abs(WF))
        one_by_norm_WF = torch.divide(torch.tensor(1.0), norm_WF)
        one_by_norm_WF = one_by_norm_WF.nan_to_num_()
        return torch.einsum("i,ijk->ijk", one_by_norm_WF, WF)

    def eval_grad_s_WF(self, parameter: torch.Tensor) -> torch.Tensor:
        if not self.flag_norm:
            return self.eval_grad_s(parameter)
        WF = self.eval(parameter)
        norm_WF = trapzInt2D(torch.abs(WF))
        one_by_norm_WF = torch.divide(torch.tensor(1.0), norm_WF)
        one_by_norm_WF = one_by_norm_WF.nan_to_num_()
        return torch.einsum("i,ijkl->ijkl", one_by_norm_WF, self.eval_grad_s(parameter))

    def apply_know_values(self, parameter: torch.Tensor) -> torch.Tensor:
        if not self.flag_given_values:
            return parameter

        return parameter @ self.ParameterToNbfsMap + self.applied_given_values
