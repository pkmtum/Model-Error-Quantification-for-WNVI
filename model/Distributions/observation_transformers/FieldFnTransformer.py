import torch
from model.Distributions.observation_transformers.BaseObservationTransformer import BaseObservationTransformer
from model.ParentClasses.class_ParameterToField import ParameterToField
from model.ParentClasses.class_nParametersToFields import nParametersToFields
from model.Distributions.observation_transformers.Observation_utils import input_dict_for_ParameterToField

class FieldFnTransformer(BaseObservationTransformer):
    def __init__(self, OutputName: str, 
                 transformerInputKeys: str | list[str] | tuple[str],
                 FieldTransformerFunction: callable) -> None:
        super().__init__(OutputName)
        self.transformerInputKeys = self._make_to_ordered(transformerInputKeys)
        self.FieldTransformerFunction = FieldTransformerFunction

    def apply(self, **kwargs) -> torch.Tensor:
        parameters = input_dict_for_ParameterToField(self.transformerInputKeys, **kwargs)
        return self.FieldTransformerFunction(**parameters)