def create_observation_transformer(obs_cfg):
    if obs_cfg is None:
        return None
    obs_type = obs_cfg["type"]
    if obs_type == "jump":
        from model.Distributions.observation_transformers.JumpTransformer import (
            JumpTransformer,
        )

        return JumpTransformer(**{k: v for k, v in obs_cfg.items() if k != "type"})
    elif obs_type == "static_conservation_residual":
        from model.Distributions.observation_transformers.ConservationResidualTransformer import (
            ConservationResidualTransformer,
        )

        return ConservationResidualTransformer(
            **{k: v for k, v in obs_cfg.items() if k != "type"}
        )
    elif obs_type == "harmonic_conservation_residual":
        from model.Distributions.observation_transformers.HarmonicConservationResidualTransformer import (
            HarmonicConservationResidualTransformer,
        )

        return HarmonicConservationResidualTransformer(
            **{k: v for k, v in obs_cfg.items() if k != "type"}
        )
    elif obs_type == "field_fn":
        from model.Distributions.observation_transformers.FieldFnTransformer import (
            FieldFnTransformer,
        )

        return FieldFnTransformer(**{k: v for k, v in obs_cfg.items() if k != "type"})
    elif obs_type == "constitutive_residual":
        from model.Distributions.observation_transformers.ConstitutiveResidualTransformer import (
            ConstitutiveResidualTransformer,
        )

        return ConstitutiveResidualTransformer(
            **{k: v for k, v in obs_cfg.items() if k != "type"}
        )
    else:
        raise ValueError(f"Unknown observation transformer type: {obs_type}")
