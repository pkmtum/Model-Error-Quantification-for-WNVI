from model.Distributions.delta_distributions.base_delta_distribution import (
    DeltaDistribution,
)
from model.Distributions.observation_transformers.observation_transformer_factory import (
    create_observation_transformer,
)
from model.Distributions.mean_functions.mean_function_factory import MeanFunctionFactory


def delta_distribution_factory(config):
    name = config["name"]
    mean_cfg = config["mean"]

    # Mean function
    mean_fn = MeanFunctionFactory(mean_cfg)

    # Observation transformer (optional)
    obs_fn = create_observation_transformer(config.get("observation_transformer", None))

    return DeltaDistribution(name, mean_fn, observed_fn=obs_fn)
