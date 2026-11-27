from model.Distributions.normal_distributions.base_normal_distribution import (
    NormalDistribution,
)
from model.Distributions.observation_transformers.observation_transformer_factory import (
    create_observation_transformer,
)
from model.Distributions.mean_functions.mean_function_factory import MeanFunctionFactory


def distribution_factory(config):
    name = config["name"]
    mean_cfg = config["mean"]
    cov_cfg = config["covariance"]

    # Mean function
    mean_fn = MeanFunctionFactory(mean_cfg)

    # Covariance function
    if cov_cfg["type"] == "diagonal":
        from model.Distributions.normal_distributions.covariance_functions.diagonal_covariance import (
            DiagonalCovariance,
        )

        cov_fn = DiagonalCovariance(**{k: v for k, v in cov_cfg.items() if k != "type"})
    elif cov_cfg["type"] == "input_diagonal_precision":
        from model.Distributions.normal_distributions.covariance_functions.input_diagonal_precision import (
            InputDiagonalPrecision,
        )

        cov_fn = InputDiagonalPrecision(
            **{k: v for k, v in cov_cfg.items() if k != "type"}
        )
    elif cov_cfg["type"] == "reduced":
        from model.Distributions.normal_distributions.covariance_functions.reduced_covariance import (
            ReducedCovariance,
        )

        cov_fn = ReducedCovariance(**{k: v for k, v in cov_cfg.items() if k != "type"})
    elif cov_cfg["type"] == "reduced_reparameterized":
        from model.Distributions.normal_distributions.covariance_functions.reduced_covariance_reparameterized import (
            ReducedCovarianceReparameterized,
        )

        cov_fn = ReducedCovarianceReparameterized(
            **{k: v for k, v in cov_cfg.items() if k != "type"}
        )
    elif cov_cfg["type"] == "constant_diagonal":
        from model.Distributions.normal_distributions.covariance_functions.constant_diagonal_covariance import (
            ConstantDiagonalCovariance,
        )

        cov_fn = ConstantDiagonalCovariance(
            **{k: v for k, v in cov_cfg.items() if k != "type"}
        )
    elif cov_cfg["type"] == "constant_scalar":
        from model.Distributions.normal_distributions.covariance_functions.constant_scalar_covariance import (
            ConstantScalarCovariance,
        )

        cov_fn = ConstantScalarCovariance(
            **{k: v for k, v in cov_cfg.items() if k != "type"}
        )
    else:
        raise ValueError(f"Unknown covariance type: {cov_cfg['type']}")

    # Observation transformer (optional)
    obs_fn = create_observation_transformer(config.get("observation_transformer", None))

    return NormalDistribution(name, mean_fn, cov_fn, observed_fn=obs_fn)
