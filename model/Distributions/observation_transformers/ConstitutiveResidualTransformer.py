import torch
from model.Distributions.observation_transformers.BaseObservationTransformer import (
    BaseObservationTransformer,
)
from model.Distributions.observation_transformers.Observation_utils import (
    input_dict_for_ParameterToField,
)
from typing import Callable


class ConstitutiveResidualTransformer(BaseObservationTransformer):
    def __init__(
        self,
        OutputName: str,
        transformerInputKeysStress: str | list[str] | tuple[str, ...],
        transformerInputKeysConstitutive: str | list[str] | tuple[str, ...],
        ConstitutiveModel_fn: Callable[..., torch.Tensor],
        ParameterToStress_fn: Callable[..., torch.Tensor],
        flag_one_residual_per_parameter: bool = False,
        flag_no_norm: bool = False,
    ) -> None:
        super().__init__(OutputName)
        self.transformerInputKeysStress = self._make_to_ordered(
            transformerInputKeysStress
        )
        self.transformerInputKeysConstitutive = self._make_to_ordered(
            transformerInputKeysConstitutive
        )
        self.ConstitutiveModel_fn = ConstitutiveModel_fn
        self.ParameterToStress_fn = ParameterToStress_fn

        # Helper to make [..., 2 x 2] to [..., 3]
        # Calling tensor[0, 1, 3] can only be performed on CPU, thus this way.
        DimensionCorretor = torch.zeros(2, 2, 3)
        DimensionCorretor[0, 0, 0] = 1.0  # sigma_xx
        DimensionCorretor[0, 1, 1] = 1.0  # sigma_xy
        DimensionCorretor[1, 1, 2] = 1.0  # sigma_yy
        self.DimensionCorretor = DimensionCorretor

        self.flag_one_residual_per_parameter = flag_one_residual_per_parameter
        self.flag_no_norm = flag_no_norm

    def apply(self, **kwargs) -> torch.Tensor:
        ConstitutiveModelParameters = dict(
            (k, kwargs[k]) for k in self.transformerInputKeysConstitutive if k in kwargs
        )
        StressParameters = input_dict_for_ParameterToField(
            self.transformerInputKeysStress, **kwargs
        )

        realStress = self.ParameterToStress_fn(**StressParameters)
        constitutiveStress = self.ConstitutiveModel_fn(**ConstitutiveModelParameters)
        if realStress.size()[-2:] == torch.Size([2, 2]):
            realStress = torch.einsum(
                "...ij,ijk->...k", realStress, self.DimensionCorretor
            )
        if constitutiveStress.size()[-2:] == torch.Size([2, 2]):
            constitutiveStress = torch.einsum(
                "...ij,ijk->...k", constitutiveStress, self.DimensionCorretor
            )
        res_3_entries = realStress - constitutiveStress

        if self.flag_one_residual_per_parameter:
            if self.flag_no_norm:
                r_c = res_3_entries.flatten(start_dim=-2)
            else:
                r_c = torch.pow(
                    res_3_entries.flatten(start_dim=-2), 2
                )  # [... x Nele * 3 x Nq]
        else:
            if self.flag_no_norm:
                r_c = torch.sum(res_3_entries, dim=(-1))  # [... x Nele x Nq]
            else:
                r_c = torch.norm(res_3_entries, dim=(-1), p=2)  # [... x Nele x Nq]

        # # when we deal with samples that have quadrature points Nq
        # if r_c.dim() == 3:
        #     r_c = torch.sum(r_c, dim=-1)

        return r_c
