from model.Distributions.mean_functions.base_mean_function import MeanFunction


def MeanFunctionFactory(mean_cfg: dict) -> MeanFunction:
    """
    Factory function to create a mean function based on the specified kind.

    Args:
        kind (str): The type of mean function to create.
        **kwargs: Additional parameters for the mean function.

    Returns:
        MeanFunction: An instance of the specified mean function type.
    """
    # Mean function
    if mean_cfg["type"] == "learnable":
        from model.Distributions.mean_functions.learnable_mean import (
            LearnableMean,
        )

        mean_fn = LearnableMean(**{k: v for k, v in mean_cfg.items() if k != "type"})
    elif mean_cfg["type"] == "neural_net":
        from model.Distributions.mean_functions.neural_net_mean import (
            NeuralNetMean,
        )

        mean_fn = NeuralNetMean(**{k: v for k, v in mean_cfg.items() if k != "type"})
    elif mean_cfg["type"] == "res_net":
        from model.Distributions.mean_functions.res_net_mean import (
            ResNetMean,
        )

        mean_fn = ResNetMean(**{k: v for k, v in mean_cfg.items() if k != "type"})
    elif mean_cfg["type"] == "constant_scalar":
        from model.Distributions.mean_functions.constant_scalar_mean import (
            ConstantScalarMean,
        )

        mean_fn = ConstantScalarMean(
            **{k: v for k, v in mean_cfg.items() if k != "type"}
        )
    elif mean_cfg["type"] == "constant_vector":
        from model.Distributions.mean_functions.constant_vector_mean import (
            ConstantVectorMean,
        )

        mean_fn = ConstantVectorMean(
            **{k: v for k, v in mean_cfg.items() if k != "type"}
        )
    elif mean_cfg["type"] == "neural_net_with_function":
        from model.Distributions.mean_functions.neural_net_mean_with_function import (
            NeuralNetMeanWithFunction,
        )

        mean_fn = NeuralNetMeanWithFunction(
            **{k: v for k, v in mean_cfg.items() if k != "type"}
        )
    elif mean_cfg["type"] == "res_net_with_function":
        from model.Distributions.mean_functions.res_net_mean_with_function import (
            ResNetMeanWithFunction,
        )

        mean_fn = ResNetMeanWithFunction(
            **{k: v for k, v in mean_cfg.items() if k != "type"}
        )
    elif mean_cfg["type"] == "res_add_with_function":
        from model.Distributions.mean_functions.res_add_mean_with_function import (
            ResAddWithFunction,
        )

        mean_fn = ResAddWithFunction(
            **{k: v for k, v in mean_cfg.items() if k != "type"}
        )
    else:
        raise ValueError(f"Unknown mean type: {mean_cfg['type']}")
    return mean_fn
