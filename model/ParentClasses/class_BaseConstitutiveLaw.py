import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseConstitutiveLaw(ABC):
    """Base class for constitutive laws that calculate stress parameters."""

    def __init__(self, options: Dict[str, Any], required_inputs: List[str]):
        """
        Initialize the constitutive law.

        Args:
            options: Dictionary containing material parameters and field mappings
            required_inputs: List of parameter names required for calculate_stress method
        """
        self.options = options
        self.required_inputs = required_inputs
        self._validate_required_options()

    @abstractmethod
    def _validate_required_options(self) -> None:
        """Validate that required options are present in the options dict."""
        pass

    @abstractmethod
    def calculate_stress(self, **kwargs) -> torch.Tensor:
        """
        Calculate stress tensor from input parameters.

        Args:
            **kwargs: Variable arguments - must contain all parameters in self.required_inputs

        Returns:
            Stress tensor of shape [..., n_elements, dim, dim]

        Raises:
            ValueError: If any required input is missing from kwargs
        """
        # Validate that all required inputs are provided
        missing_inputs = [inp for inp in self.required_inputs if inp not in kwargs]
        if missing_inputs:
            raise ValueError(f"Missing required inputs: {missing_inputs}")

    def get_material_properties(self) -> Dict[str, Any]:
        """Return material properties for this constitutive law."""
        return {
            key: value
            for key, value in self.options.items()
            if not hasattr(value, "__call__")
        }

    def get_required_inputs(self) -> List[str]:
        """Return list of required input parameter names."""
        return self.required_inputs.copy()
