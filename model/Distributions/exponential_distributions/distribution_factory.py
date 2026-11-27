from model.Distributions.exponential_distributions.base_exponential_distribution import (
    ExponentialDistribution,
)
from model.Distributions.observation_transformers.observation_transformer_factory import (
    create_observation_transformer,
)


def exponential_distribution_factory(config):
    """
    Factory function to create exponential distributions with different rate functions.

    Args:
        config: Configuration dictionary with keys:
            - name: Distribution name (prefix)
            - rate: Rate function configuration
            - observation_transformer: Optional observation transformer config

    Returns:
        ExponentialDistribution: Configured exponential distribution
    """
    name = config["name"]
    rate_cfg = config["rate"]

    # Create rate function based on type
    if rate_cfg["type"] == "constant":
        from model.Distributions.exponential_distributions.rate_functions.constant_rate import (
            ConstantRate,
        )

        rate_fn = ConstantRate(**{k: v for k, v in rate_cfg.items() if k != "type"})

    elif rate_cfg["type"] == "learnable_vector":
        from model.Distributions.exponential_distributions.rate_functions.learnable_vector_rate import (
            LearnableVectorRate,
        )

        rate_fn = LearnableVectorRate(
            **{k: v for k, v in rate_cfg.items() if k != "type"}
        )

    elif rate_cfg["type"] == "input":
        from model.Distributions.exponential_distributions.rate_functions.input_rate import (
            InputRate,
        )

        rate_fn = InputRate(**{k: v for k, v in rate_cfg.items() if k != "type"})

    else:
        raise ValueError(f"Unknown rate type: {rate_cfg['type']}")

    # Observation transformer (optional)
    obs_fn = create_observation_transformer(config.get("observation_transformer", None))

    return ExponentialDistribution(name, rate_fn, observed_fn=obs_fn)
