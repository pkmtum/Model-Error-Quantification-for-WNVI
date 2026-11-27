import torch.nn as nn
from abc import ABC, abstractmethod
import torch


class CovarianceFunction(nn.Module, ABC):
    def __init__(
        self,
        dim: int,
        std: float | torch.Tensor,
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

    @abstractmethod
    def sample(self, mean, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def log_prob(self, x, mean, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def entropy(self, **kwargs) -> torch.Tensor:
        pass

    def get_sample_shape(
        self, num_samples: int | None, mean_shape: torch.Size | tuple[int, ...]
    ) -> torch.Size:
        # we want to ensure that the output is always 2D
        D = mean_shape[-1] if isinstance(mean_shape, torch.Size) else mean_shape[-1]
        B = mean_shape[0] if len(mean_shape) > 1 else 1
        if num_samples is None:
            num_samples = 1
        if B == 1:
            return torch.Size([num_samples, D])
        else:
            return torch.Size([B, D])
