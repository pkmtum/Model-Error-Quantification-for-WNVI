import torch
from model.Distributions.observation_transformers.BaseObservationTransformer import BaseObservationTransformer


class ConservationResidualTransformer(BaseObservationTransformer):
    def __init__(self, 
                 OutputName: str,
                 ThetaDataLoader: torch.utils.data.DataLoader, 
                 transformerInputKeys:  str | list[str] | tuple[str], 
                 ResidualFunction: callable) -> None:
        
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

        # Residual calculation for multiple (or one) theta
        res = self.ResidualFunction(theta=theta, **parameters)

        return res.T # I want first number of (xi) samples and then number of thetas 
