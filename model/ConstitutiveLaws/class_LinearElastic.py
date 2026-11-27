import torch
from model.ParentClasses.class_BaseConstitutiveLaw import BaseConstitutiveLaw
from typing import Dict, Any


class LinearElastic(BaseConstitutiveLaw):
    """Linear elastic constitutive law implementing Hooke's law."""

    def __init__(self, options: Dict[str, Any]):
        """
        Initialize linear elastic material.

        Required options:
            - nu: Poisson's ratio (float)
            - XToField: Material field mapper for Young's modulus
            - YToField: Displacement field mapper

        Optional options:
            - flag_E_plus_one: Whether to add 1 to Young's modulus (bool, default False)
        """
        # Define which inputs are required for calculate_stress method
        required_inputs = ["x", "y"]
        super().__init__(options, required_inputs)

        self.nu = self.options["nu"]
        self.XToField = self.options["XToField"]
        self.YToField = self.options["YToField"]
        self.flag_E_plus_one = self.options.get("flag_E_plus_one", False)

    def _validate_required_options(self) -> None:
        """Validate required options for linear elastic material."""
        required_options = ["nu", "XToField", "YToField"]
        missing_options = [key for key in required_options if key not in self.options]
        if missing_options:
            raise ValueError(f"Required options missing: {missing_options}")

    def calculate_stress(self, **kwargs) -> torch.Tensor:
        """
        Calculate stress using linear elastic constitutive law.

        Required inputs:
            x: Material parameters (log Young's modulus)
            y: Displacement parameters

        Returns:
            Stress tensor of shape [..., n_elements, 2, 2]
        """
        # Call parent validation first
        super().calculate_stress(**kwargs)

        # Extract the required inputs
        x = kwargs["x"]
        y = kwargs["y"]

        # Transform material parameters: log(E) -> E
        x_transformed = torch.sigmoid(x) * 5.0 - 1.0  # normalize to [0, 10]
        x_transformed = torch.exp(x_transformed)
        E = self.XToField.eval(x_transformed)  # [..., n_elements]

        if self.flag_E_plus_one:
            E = E + torch.tensor(1.0)

        # Calculate Lamé constants
        mu = E / (2 * (1 + self.nu))  # [..., n_elements]
        lmbda = E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))  # [..., n_elements]

        # Get displacement gradients
        du = self.YToField.eval_grad_s(y)  # [..., n_elements, dim_u, dim_s]

        # Calculate strain tensor: ε = 0.5(∇u + ∇u^T)
        eps = 0.5 * (torch.einsum("...ijk->...ikj", du) + du)  # [..., n_elements, 2, 2]

        # Calculate trace of strain tensor
        tr_eps = torch.einsum("...ijj->...i", eps)  # [..., n_elements]

        # First term: 2μϵ
        summand_1 = 2 * torch.einsum(
            "...i,...ijk->...ijk", mu, eps
        )  # [..., n_elements, 2, 2]

        # Second term: λ tr(ϵ) I
        factor_1 = lmbda * tr_eps  # [..., n_elements]
        summand_2 = torch.einsum(
            "kl,...i->...ikl", torch.eye(2), factor_1
        )  # [..., n_elements, 2, 2]

        # Cauchy stress tensor
        sigma = summand_1 + summand_2  # [..., n_elements, 2, 2]

        return sigma

    def get_lame_constants(self, E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate Lamé constants from Young's modulus.

        Args:
            E: Young's modulus tensor

        Returns:
            Tuple of (mu, lambda) Lamé constants
        """
        mu = E / (2 * (1 + self.nu))
        lmbda = E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        return mu, lmbda
