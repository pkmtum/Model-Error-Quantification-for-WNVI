import torch
from abc import ABC, abstractmethod


class BasePDE(ABC):
    """Base class for PDEs."""

    def __init__(self, ThetaToField, args_rhs: dict, options: dict):
        self.ThetaToField = ThetaToField
        self.args_rhs = args_rhs
        self.options = options

    def is_harmonic(self) -> bool:
        """Check if this is a harmonic PDE that needs displacement parameters."""
        return (
            hasattr(self, "YToField")
            and hasattr(self, "rho")
            and hasattr(self, "omega")
        )

    @abstractmethod
    def forward(self, xi: torch.Tensor, theta: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Solve the PDE given stress parameters xi and weight function parameters theta.

        Args:
            xi: Stress parameters
            theta: Weight function parameters
            **kwargs: Additional parameters specific to different PDE types (e.g., y for harmonic)

        Returns:
            Residual of the PDE
        """
        pass

    @abstractmethod
    def _lhs(self, xi: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Left-hand side of the weak form."""
        pass

    @abstractmethod
    def _rhs(self, theta: torch.Tensor) -> torch.Tensor:
        """Right-hand side of the weak form."""
        pass
