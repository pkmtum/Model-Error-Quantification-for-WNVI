import torch.nn as nn
import torch
from abc import ABC, abstractmethod


class MeanFunction(nn.Module, ABC):
    def __init__(
        self,
        value: float | list[float] | torch.Tensor | str,
        dim: int,
        special_inputs: list[str] | str | None = None,
    ) -> None:
        super().__init__()
        self.special_inputs = (
            special_inputs
            if isinstance(special_inputs, list)
            else [special_inputs]
            if special_inputs
            else []
        )
        self.dim = dim

    @abstractmethod
    def compute(self, **kwargs) -> torch.Tensor:
        pass
