from model.Distributions.gamma_distributions.base_gamma_function import (
    ModularGammaDistribution,
)


def gamma_distribution_factory(config):
    prefix = config.get("name", "x")
    gamma_cfg = config["gamma"]

    if gamma_cfg["type"] == "constant_scalar":
        from model.Distributions.gamma_distributions.gamma_functions.constant_scalar_gamma import (
            ConstantScalarGamma,
        )

        gamma_fn = ConstantScalarGamma(
            **{k: v for k, v in gamma_cfg.items() if k != "type"}
        )
    elif gamma_cfg["type"] == "constant_vector":
        from model.Distributions.gamma_distributions.gamma_functions.constant_vector_gamma import (
            ConstantVectorGamma,
        )

        gamma_fn = ConstantVectorGamma(
            **{k: v for k, v in gamma_cfg.items() if k != "type"}
        )
    elif gamma_cfg["type"] == "learnable_vector":
        from model.Distributions.gamma_distributions.gamma_functions.learnable_vector_gamma import (
            LearnableVectorGamma,
        )

        gamma_fn = LearnableVectorGamma(
            **{k: v for k, v in gamma_cfg.items() if k != "type"}
        )
    elif gamma_cfg["type"] == "constant_vector_closed_form":
        from model.Distributions.gamma_distributions.gamma_functions.constant_vector_gamma_closed_form import (
            ConstantVectorGammaClosedForm,
        )

        gamma_fn = ConstantVectorGammaClosedForm(
            **{k: v for k, v in gamma_cfg.items() if k != "type"}
        )
    else:
        raise ValueError(f"Unknown gamma type: {gamma_cfg['type']}")

    # Observation transformer (optional)
    obs_fn = None
    from model.Distributions.observation_transformers.observation_transformer_factory import (
        create_observation_transformer,
    )

    obs_fn = create_observation_transformer(config.get("observation_transformer", None))

    return ModularGammaDistribution(prefix, gamma_fn, observed_fn=obs_fn)
