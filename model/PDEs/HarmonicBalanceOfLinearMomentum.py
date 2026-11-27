import torch
from model.ParentClasses.class_BasePDE import BasePDE


class HarmonicBalanceOfLinearMomentum(BasePDE):
    """Balance of linear momentum PDE implementation."""

    def __init__(
        self,
        ThetaToField,
        XiToField,
        YToField,
        args_rhs: dict,
        options: dict,
        rho: float = 1e-6,
        omega: float = 2 * 3.14159 * 50,
    ):
        super().__init__(ThetaToField, args_rhs, options)
        self.XiToField = XiToField
        self.YToField = YToField
        self.rho = rho
        self.omega = omega
        self._setup_neumann_bcs()

    def _setup_neumann_bcs(self):
        """Setup Neumann boundary conditions (from original __init__)."""
        # Left boundary
        if "neumann_left" in self.args_rhs.keys():
            self.left_node_matrix_0, self.flag_left_0 = self.create_precalc_matrix(
                self.ThetaToField.left_node_list,
                self.args_rhs["neumann_left"][0],
                self.ThetaToField.len_y,
            )
            self.left_node_matrix_1, self.flag_left_1 = self.create_precalc_matrix(
                self.ThetaToField.left_node_list,
                self.args_rhs["neumann_left"][1],
                self.ThetaToField.len_y,
            )
        else:
            self.left_node_matrix_0 = self.left_node_matrix_1 = torch.zeros_like(
                self.ThetaToField.left_node_list
            )
            self.flag_left_0 = self.flag_left_1 = False

        # Right boundary
        if "neumann_right" in self.args_rhs.keys():
            self.right_node_matrix_0, self.flag_right_0 = self.create_precalc_matrix(
                self.ThetaToField.right_node_list,
                self.args_rhs["neumann_right"][0],
                self.ThetaToField.len_y,
            )
            self.right_node_matrix_1, self.flag_right_1 = self.create_precalc_matrix(
                self.ThetaToField.right_node_list,
                self.args_rhs["neumann_right"][1],
                self.ThetaToField.len_y,
            )
        else:
            self.right_node_matrix_0 = self.right_node_matrix_1 = torch.zeros_like(
                self.ThetaToField.right_node_list
            )
            self.flag_right_0 = self.flag_right_1 = False

        # Bottom boundary
        if "neumann_bottom" in self.args_rhs.keys():
            self.bottom_node_matrix_0, self.flag_bottom_0 = self.create_precalc_matrix(
                self.ThetaToField.bottom_node_list,
                self.args_rhs["neumann_bottom"][0],
                self.ThetaToField.len_x,
            )
            self.bottom_node_matrix_1, self.flag_bottom_1 = self.create_precalc_matrix(
                self.ThetaToField.bottom_node_list,
                self.args_rhs["neumann_bottom"][1],
                self.ThetaToField.len_x,
            )
        else:
            self.bottom_node_matrix_0 = self.bottom_node_matrix_1 = torch.zeros_like(
                self.ThetaToField.bottom_node_list
            )
            self.flag_bottom_0 = self.flag_bottom_1 = False

        # Top boundary
        if "neumann_top" in self.args_rhs.keys():
            self.top_node_matrix_0, self.flag_top_0 = self.create_precalc_matrix(
                self.ThetaToField.top_node_list,
                self.args_rhs["neumann_top"][0],
                self.ThetaToField.len_x,
            )
            self.top_node_matrix_1, self.flag_top_1 = self.create_precalc_matrix(
                self.ThetaToField.top_node_list,
                self.args_rhs["neumann_top"][1],
                self.ThetaToField.len_x,
            )
        else:
            self.top_node_matrix_0 = self.top_node_matrix_1 = torch.zeros_like(
                self.ThetaToField.top_node_list
            )
            self.flag_top_0 = self.flag_top_1 = False

    def create_precalc_matrix(
        self,
        node_list: torch.Tensor,
        neumann_value: float | torch.Tensor,
        len_dim: float | torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """Create precalculated matrix for Neumann boundary conditions."""
        if torch.allclose(torch.tensor(neumann_value), torch.tensor(0.0)):
            return torch.zeros_like(node_list), False
        node_matrix = node_list * neumann_value * len_dim * 0.5
        return node_matrix, True

    def _lhs(self, xi: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Left-hand side: integral of stress and test function gradients."""
        sigma = self._stress(xi)
        dw = self.ThetaToField.eval_grad_s(theta)
        if dw.dim() == 3:
            dw = dw.unsqueeze(0)
        if sigma.dim() == 3:
            sigma = sigma.unsqueeze(0)
        eps_w = 0.5 * (torch.einsum("...ijk->...ikj", dw) + dw)
        integrant_i = torch.einsum("hikj,...ijk->...hi", sigma, eps_w)
        integral_i = torch.einsum(
            "i, ...i->...i", self.ThetaToField.Two_A * 0.5, integrant_i
        )
        integral = torch.sum(integral_i, dim=-1)
        return integral

    def _stress(self, xi: torch.Tensor) -> torch.Tensor:
        """Get stress from stress parameters xi."""
        sigma = self.XiToField.eval(xi)
        return sigma

    def _rhs(self, theta: torch.Tensor) -> torch.Tensor:
        """Right-hand side: Neumann boundary conditions and body forces."""
        neumann = self._neumann(theta)
        return neumann

    def _neumann(self, theta: torch.Tensor) -> torch.Tensor:
        """Calculate Neumann boundary condition contributions."""
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        theta = self.ThetaToField.apply_know_values(theta)
        theta_middle = theta.size(dim=1) // 2
        theta_x = theta[:, :theta_middle]
        theta_y = theta[:, theta_middle:]
        neumann = torch.zeros(theta.size(dim=0))

        if self.flag_left_0:
            neumann += torch.matmul(theta_x, self.left_node_matrix_0)
        if self.flag_left_1:
            neumann += torch.matmul(theta_y, self.left_node_matrix_1)
        if self.flag_right_0:
            neumann += torch.matmul(theta_x, self.right_node_matrix_0)
        if self.flag_right_1:
            neumann += torch.matmul(theta_y, self.right_node_matrix_1)
        if self.flag_bottom_0:
            neumann += torch.matmul(theta_x, self.bottom_node_matrix_0)
        if self.flag_bottom_1:
            neumann += torch.matmul(theta_y, self.bottom_node_matrix_1)
        if self.flag_top_0:
            neumann += torch.matmul(theta_x, self.top_node_matrix_0)
        if self.flag_top_1:
            neumann += torch.matmul(theta_y, self.top_node_matrix_1)
        return neumann

    def _lhs_with_stress(
        self, stress: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        """Left-hand side using direct stress tensor instead of xi parameters."""
        sigma = stress
        dw = self.ThetaToField.eval_grad_s(theta)
        if dw.dim() == 3:
            dw = dw.unsqueeze(0)
        if sigma.dim() == 3:
            sigma = sigma.unsqueeze(0)
        eps_w = 0.5 * (torch.einsum("...ijk->...ikj", dw) + dw)
        integrant_i = torch.einsum("hikj,...ijk->...hi", sigma, eps_w)
        integral_i = torch.einsum(
            "i, ...i->...i", self.ThetaToField.Two_A * 0.5, integrant_i
        )
        integral = torch.sum(integral_i, dim=-1)
        return integral

    def forward(self, xi: torch.Tensor, theta: torch.Tensor, **kwargs) -> torch.Tensor:
        """Solve the balance of linear momentum equation."""
        # Extract y from kwargs for harmonic terms
        y = kwargs.get("y", None)
        if y is None:
            raise ValueError(
                "HarmonicBalanceOfLinearMomentum requires 'y' parameter in kwargs"
            )

        lhs = self._lhs(xi, theta)
        lhs2 = self._lhs2(y, theta)
        rhs = self._rhs(theta)
        if lhs.dim() == 2:
            rhs = rhs.unsqueeze(1)
        res_unnormalized = lhs - lhs2 - rhs
        # weight_function_area = (
        #     (torch.abs(theta).sum(dim=-1) + 1) * self.ThetaToField.Two_A[0] / 2
        # )
        # residual = res_unnormalized / weight_function_area.unsqueeze(-1)
        return res_unnormalized

    def forward_with_stress(
        self, stress: torch.Tensor, theta: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Solve the balance of linear momentum equation using direct stress tensor."""
        # Extract y from kwargs for harmonic terms
        y = kwargs.get("y", None)
        if y is None:
            raise ValueError(
                "HarmonicBalanceOfLinearMomentum requires 'y' parameter in kwargs"
            )

        lhs = self._lhs_with_stress(stress, theta)
        lhs2 = self._lhs2(y, theta)
        rhs = self._rhs(theta)
        if lhs.dim() == 2:
            rhs = rhs.unsqueeze(1)
        res_unnormalized = lhs - lhs2 - rhs
        # weight_function_area = (
        #     (torch.abs(theta).sum(dim=-1) + 1) * self.ThetaToField.Two_A[0] / 2
        # )
        # residual = res_unnormalized / weight_function_area.unsqueeze(-1)
        return res_unnormalized

    def _lhs2(self, y, theta):
        constant = self.rho * self.omega**2
        # calculate the values at the quadrature points
        u_Q = self.YToField.eval_quadrature_points(y)
        w_Q = self.ThetaToField.eval_quadrature_points(theta)
        # dot product of u_Q and w_Q
        u_dot_w_Q = torch.einsum("unij,wnij->uwni", u_Q, w_Q)  # [... x Nele x Q]
        # sum over Q
        u_dot_w = torch.sum(u_dot_w_Q, dim=-1)  # [... x Nele]
        # multiply with constant and weight (here 1/6 for all Q) and elementwise the Jacobian (here 2*A)
        integral_i = (
            torch.tensor(1 / 6)
            * constant
            * torch.einsum("i,...i->...i", self.ThetaToField.Two_A, u_dot_w)
        )
        # sum over all elements
        integral = torch.sum(integral_i, dim=-1)  # [...]
        return integral.T
