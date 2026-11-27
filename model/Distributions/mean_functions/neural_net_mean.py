import torch
from model.Distributions.mean_functions.base_mean_function import MeanFunction
from utils.nn_funcs.function_FFNBuilder import build_ffn


class NeuralNetMean(MeanFunction):
    def __init__(
        self,
        input_names,
        input_dims,
        dim,  # denotes the neural network output dimension
        activation_func_name="ReLU",
        hidden_layers=(64,),
        flagLayerNormalization=False,
    ):
        super().__init__(value=0.0, dim=dim, special_inputs=input_names)
        input_dim_total = sum(input_dims)
        self.network = build_ffn(
            input_dim_total,
            dim,
            activation_func_name,
            hidden_layers,
            flagLayerNormalization,
        )

    def compute(self, num_samples=None, **kwargs):
        inputs = [kwargs[name] for name in self.special_inputs]
        # Flatten all but the first (batch) dimension for each input
        flat_inputs = [i.reshape(i.shape[0], -1) if i.dim() > 2 else i for i in inputs]
        z = torch.cat(flat_inputs, dim=-1)
        out = self.network(z)
        # num_samples is ignored
        return out
