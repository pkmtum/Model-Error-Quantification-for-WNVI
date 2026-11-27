def distribution_factory_selection(kind, config):
    """
    Selects and calls the correct distribution factory based on the kind.
    Args:
        kind (str): The type of distribution ("normal", "delta", "gamma", or "exponential").
        config (dict): The configuration dictionary for the distribution.
    Returns:
        Distribution instance.
    """
    if kind.lower() == "normal":
        from model.Distributions.normal_distributions.distribution_factory import (
            distribution_factory,
        )

        return distribution_factory(config)
    elif kind.lower() == "delta":
        from model.Distributions.delta_distributions.distribution_factory import (
            delta_distribution_factory,
        )

        return delta_distribution_factory(config)
    elif kind.lower() == "gamma":
        from model.Distributions.gamma_distributions.distribution_factory import (
            gamma_distribution_factory,
        )

        return gamma_distribution_factory(config)
    elif kind.lower() == "exponential":
        from model.Distributions.exponential_distributions.distribution_factory import (
            exponential_distribution_factory,
        )

        return exponential_distribution_factory(config)
    else:
        raise ValueError(f"Unknown distribution kind: {kind}")
