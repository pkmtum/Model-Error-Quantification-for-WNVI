import torch
from model.Distributions.mean_functions.base_mean_function import MeanFunction
from utils.nn_funcs.function_FFNBuilder import build_ffn
from typing import Callable


class ResNetMeanWithFunction(MeanFunction):
    def __init__(
        self,
        input_names,
        input_dims,
        dim,  # denotes the neural network output dimension
        transformation_fn: Callable[..., torch.Tensor],
        activation_func_name="ReLU",
        hidden_layers=(64,),
        flagLayerNormalization=False,
    ):
        super().__init__(value=0.0, dim=dim, special_inputs=input_names)
        input_dim_total = sum(input_dims)
        self.input_dim_total = input_dim_total
        self.network = build_ffn(
            input_dim_total,
            dim,
            activation_func_name,
            hidden_layers,
            flagLayerNormalization,
        )
        self.transformation_fn = transformation_fn

        # Helper to make [..., 2 x 2] to [..., 3]
        # Calling tensor[0, 1, 3] can only be performed on CPU, thus this way.
        DimensionCorretor = torch.zeros(2, 2, 3)
        DimensionCorretor[0, 0, 0] = 1.0  # sigma_xx
        DimensionCorretor[0, 1, 1] = 1.0  # sigma_xy
        DimensionCorretor[1, 1, 2] = 1.0  # sigma_yy
        self.DimensionCorretor = DimensionCorretor

    def compute(self, num_samples=None, **kwargs):
        transformed_inputs = self.transformation_fn(**kwargs)

        # this is currently only used for the stress calculation, and I only need 3 of 4 dimensions
        if transformed_inputs.size()[-2:] == torch.Size([2, 2]):
            transformed_inputs = torch.einsum(
                "...ij,ijk->...k", transformed_inputs, self.DimensionCorretor
            )
        # and it should be flattened to [..., Nbfs * 3]
        if transformed_inputs.size()[-1] != self.input_dim_total:
            transformed_inputs = transformed_inputs.movedim(-1, -2).flatten(
                start_dim=-2, end_dim=-1
            )

        out = self.network(transformed_inputs)

        # ResNet always uses residual connection - input dimensions must match output dimensions
        if transformed_inputs.shape[-1] != out.shape[-1]:
            raise ValueError(
                f"ResNetWithFunction requires transformed input and output dimensions to match for residual connection. "
                f"Got transformed input dim: {transformed_inputs.shape[-1]}, output dim: {out.shape[-1]}. "
                f"Set dim={transformed_inputs.shape[-1]} in the configuration."
            )

        final = (
            transformed_inputs + out
        )  # Residual connection: transformed_input + network output
        # num_samples is ignored
        return final
