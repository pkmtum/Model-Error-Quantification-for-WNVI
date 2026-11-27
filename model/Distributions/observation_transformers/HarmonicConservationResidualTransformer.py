import torch
from typing import Callable, Union, List, Tuple
from model.Distributions.observation_transformers.BaseObservationTransformer import (
    BaseObservationTransformer,
)


class HarmonicConservationResidualTransformer(BaseObservationTransformer):
    def __init__(
        self,
        OutputName: str,
        ThetaDataLoader,
        transformerInputKeys: Union[str, List[str], Tuple[str]],
        ResidualFunction: Callable,
    ) -> None:
        super().__init__(OutputName)
        self.ThetaDataLoader = ThetaDataLoader
        self.ThetaDataIterator = iter(ThetaDataLoader)
        self.transformerInputKeys = self._make_to_ordered(transformerInputKeys)
        self.ResidualFunction = ResidualFunction

    def apply(self, **kwargs) -> torch.Tensor:
        parameters = {k: kwargs[k] for k in self.transformerInputKeys if k in kwargs}

        # Get next batch of theta from loader
        try:
            batch = next(self.ThetaDataIterator)
        except StopIteration:
            self.ThetaDataIterator = iter(self.ThetaDataLoader)
            batch = next(self.ThetaDataIterator)

        # DataLoader yields tuples, so extract the tensor
        theta = batch[0] if isinstance(batch, (tuple, list)) else batch

        # Extract xi and y parameters for harmonic residual calculation
        # The harmonic balance PDE requires both xi and y as inputs
        xi = parameters.get("xi")
        y = parameters.get("y")

        if xi is None:
            raise ValueError(
                "HarmonicConservationResidualTransformer requires 'xi' parameter"
            )
        if y is None:
            raise ValueError(
                "HarmonicConservationResidualTransformer requires 'y' parameter"
            )

        # Residual calculation for multiple (or one) theta with both xi and y parameters
        # Pass y through kwargs as expected by HarmonicBalanceOfLinearMomentum.forward()
        res = self.ResidualFunction(xi=xi, theta=theta, y=y)

        return res.T  # I want first number of (xi,y) samples and then number of thetas
