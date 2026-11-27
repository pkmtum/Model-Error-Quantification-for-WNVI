import torch
from model.Distributions.observation_transformers.BaseObservationTransformer import BaseObservationTransformer
from model.Distributions.observation_transformers.Observation_utils import input_dict_for_ParameterToField
from utils.torch_funcs.function_neighbours_mask import create_neighbours_mask

class JumpTransformer(BaseObservationTransformer):
    """
    A transformer that applies a jump transformation to the input data.
    This is a placeholder implementation and should be customized as needed.
    """
    
    def __init__(self, 
                 OutputName: str,
                 transformerInputKeys: str | list[str] | tuple[str],
                 FieldTransformerFunction: callable,
                 FieldNodesMap: torch.Tensor,
                 ) -> None:
        super().__init__(OutputName)
        self.transformerInputKeys = self._make_to_ordered(transformerInputKeys)
        self.FieldTransformerFunction = FieldTransformerFunction
        self.Neighbor_Mask = create_neighbours_mask(FieldNodesMap)

    def apply(self, **kwargs) -> torch.Tensor:
        """
        Apply the jump transformation to the input data.
        """
        parameters = input_dict_for_ParameterToField(
            self.transformerInputKeys, **kwargs
        )
        field = self.FieldTransformerFunction(**parameters)
        return torch.einsum('ij,...j->...i', self.Neighbor_Mask, field)
