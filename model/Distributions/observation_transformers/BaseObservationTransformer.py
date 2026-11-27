from abc import ABC, abstractmethod
from typing import Any


class BaseObservationTransformer(ABC):
    """
    Base class for observation transformers.
    """

    def __init__(self, OutputName):
        self.OutputName = OutputName

    @abstractmethod
    def apply(self, **kwargs) -> Any:
        """
        Transform the input data based on the specific implementation.
        """
        pass

    def _make_to_ordered(
        self, keys: str | list[str] | tuple[str, ...]
    ) -> tuple[str, ...]:
        """
        Convert a string or iterable of strings to a tuple of strings.
        Only accepts str, list[str], or tuple[str]. Rejects sets and other unordered types.
        """
        if isinstance(keys, str):
            return (keys,)
        elif isinstance(keys, (list, tuple)):
            if not all(isinstance(k, str) for k in keys):
                raise ValueError("All keys must be strings.")
            return tuple(keys)
        else:
            raise ValueError(
                f"Expected str, list[str], or tuple[str], got {type(keys)}"
            )
