# =============================================================================
# Brain phantom, 35 dB noise: complete two-stage experiment in ONE file.
#
# Everything below is the setup of the *second* stage (== configs/stage2.py of
# the earlier release). The `stages` list at the very bottom lists, per stage,
# which of these module-level settings are overridden. The posterior state
# (and the noisy data) are handed from stage 1 to stage 2 in memory.
# =============================================================================
seed = 0

import torch

# Grid and scale parameters
scale_x = 76
scale_y = 108

# Physical and PDE parameters
frequency = 50  # Hz
rho = 1e-6  # in kg/mm^3
nu = 0.4  # Poisson's ratio
flag_E_plus_one = False  # or False, depending on your model
MaterialModel = "LinearElastic"  # or "HarmonicHookConstitutive"

# Loading Checkpoint
flag_load_checkpoint = False  # stage hand-over happens in memory, see `stages` at the bottom
checkpoint_path = None
dont_load = ["lmbda_c"]

# Right-hand side arguments for the PDE (body force etc.)
args_rhs = {
    "neumann_top": [0.0, 0.1],
    # "neumann_right": [1.0, 0.0],
}


# Boundary condition config
bc_config = {
    "Y": {
        "apply": False,
        "conditions": [
            {"boundary": "bottom", "dim": 1, "value": 0.0},
            {"boundary": "BLCorner", "dim": 0, "value": 0.0},
        ],
    },
    "Theta": {
        "apply": True,
        "conditions": [
            {"boundary": "bottom", "dim": 1, "value": 0.0},
            {"boundary": "BLCorner", "dim": 0, "value": 0.0},
        ],
    },
}

# Options for Y, Theta, and X fields (these will be updated by create_boundary_conditions)
general_n_bfs = 64
YToField_options_init = {"n_bfs_s1": general_n_bfs, "n_bfs_s2": general_n_bfs}
ThetaToField_options_init = {"n_bfs_s1": general_n_bfs, "n_bfs_s2": general_n_bfs}
XToField_options_init = {"n_bfs_s1": general_n_bfs, "n_bfs_s2": general_n_bfs}
XiToField_options_init = {"n_bfs_s1": general_n_bfs, "n_bfs_s2": general_n_bfs}


# Example groundtruth options (dummy)
groundtruth_options = {
    # "Background": {"type": "background", "value": 2},
    # "Inclusion_1": {
    #     "type": "circular",
    #     "center_x": 9.0,
    #     "center_y": 12.5,
    #     "radius": 3.6,
    #     "value": 10,
    # },
    # "Inclusion_2": {
    #     "type": "rectangular",
    #     "x1": 0,
    #     "y1": scale_y - 3.5,
    #     "x2": 3.5,
    #     "y2": scale_y,
    #     "value": 5,
    # },
    "Transformation": ("exponential"),
}

# Device and dtype
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
default_dtype = torch.float32

# Observation grid size (example)
SNR_in_dB = 35
n_obs_s1 = 64
n_obs_s2 = 64
# Load data path
load_u_path = "data/simpler_brain_paper_64.pt"
load_stress_path = (
    "data/simpler_brain_paper_64_stress.pt"
)
load_brain_image_path = (
    "data/Brain_screenshot_simplified_2.png"
)
# load_noisy_u_path = (
#     "data/brain_paper_96_40dB.pt"
# )

# Brain material parameters
E_background = 1.0
E_gray = 1.4
E_white = 1.9

# %% Priors

prior_x_kind = "normal"  # "FieldNormal"
prior_x_options = {
    "name": "x",
    "mean": {"type": "constant_scalar", "value": 0.0},
    "covariance": {"type": "constant_scalar", "std": 2.0},
    "SpecialInput": None,
}

prior_y_kind = "normal"
prior_y_options = {
    "name": "y",
    "mean": {"type": "constant_scalar", "value": 0.0},
    "covariance": {"type": "constant_scalar", "std": 1e8},
    "SpecialInput": None,
}

prior_xi_kind = "normal"
prior_xi_options = {
    "name": "xi",
    "mean": {"type": "constant_scalar", "value": 0.0},
    "covariance": {"type": "constant_scalar", "std": 1e8},
    "SpecialInput": None,
}

prior_lmbda_c_kind = "Gamma"
prior_lmbda_c_options = {
    "name": "lmbda_c",
    "gamma": {"type": "constant_scalar", "a": 5e-7, "b": 5e-7},
    "SpecialInput": None,
}

prior_jumpPenalty_kind = "Gamma"
prior_jumpPenalty_options = {
    "name": "jumpPenalty",
    "gamma": {"type": "constant_scalar", "a": 1e-6, "b": 1e-6},
    "SpecialInput": None,
}

prior_jump_kind = "normal"
prior_jump_options = {
    "name": "jump",
    "mean": {"type": "constant_scalar", "value": 0.0},
    "covariance": {
        "type": "input_diagonal_precision",
        "precision_input_key": "jumpPenalty",
    },
    "observation_transformer": {
        "type": "jump",
        "OutputName": "jumps",
        "transformerInputKeys": "x",
    },
    "SpecialInput": ["x", "jumpPenalty"],
}

# %% Likelihoods
theta_options = [
    # {
    #     "kind": "rbf",
    #     "num_samples": 10_000,
    #     "kwargs": {"min_rad": 3.0, "max_rad": max(scale_x, scale_y) / 3},
    # },
    {"kind": "full", "num_samples": 1},
    # {
    #     "kind": "circular",
    #     "num_samples": 10_000,
    #     "kwargs": {"min_rad": 0.001, "max_rad": max(scale_x, scale_y) / 3},
    # },
]
theta_batch_size = 250

conservation_likelihood_kind = "normal"
conservation_likelihood_options = {
    "name": "conservation",
    "mean": {"type": "constant_scalar", "value": 0.0},
    "covariance": {"type": "constant_scalar", "std": 1e-4},
    "observation_transformer": {
        "type": "harmonic_conservation_residual",
        "OutputName": "r",
        "transformerInputKeys": ["xi", "y"],
    },
    "SpecialInput": ["xi", "y"],
}

constitutive_likelihood_kind = "normal"
constitutive_likelihood_options = {
    "name": "constitutive",
    "mean": {"type": "constant_scalar", "value": 0.0},
    "covariance": {
        "type": "input_diagonal_precision",
        "precision_input_key": "lmbda_c",
    },
    "observation_transformer": {
        "type": "constitutive_residual",
        "OutputName": "r_c",
        "transformerInputKeysStress": "xi",
        "transformerInputKeysConstitutive": ["x", "y"], #, "p"],
        "flag_one_residual_per_parameter": True,
        "flag_no_norm": True,
    },
    "SpecialInput": ["x", "y",  "xi", "lmbda_c"], #"p",
}

actual_likelihood_kind = "normal"
actual_likelihood_options = {
    "name": "actual",
    "mean": {"type": "constant_vector"},  # value are observed values
    "covariance": {"type": "constant_scalar"},  # std from the noise applied to the data
    "observation_transformer": {
        "type": "field_fn",
        "OutputName": "u_full",
        "transformerInputKeys": "y",
    },
    "SpecialInput": ["y"],
}

# %% Posteriors

posterior_x_kind = "normal"  # "ReducedMVN" #"MVN"#"Delta" #"CNNDelta"
posterior_x_options = {
    "name": "x",
    "mean": {
        "type": "learnable",
        "value": 0.0,
    },
    "covariance": {"type": "reduced", "reduced_dim": 10, "std": 0.01},
    "SpecialInput": [],
    "InitAtTrueValue": False,
}

posterior_y_kind = "normal"  # 'ReducedMVN' #"MVN"  # "DoubleDelta" #"DiagMVN"
posterior_y_options = {
    "name": "y",
    # "mean": {"type": "learnable", "value": 0.0},
    "mean": {
        "type": "neural_net",
        "input_names": ["x"],
        "hidden_layers": (2000, 2000, 2000),
        "activation_func_name": "SiLU",
        "flagLayerNormalization": True,
    },
    "covariance": {"type": "reduced", "reduced_dim": 10, "std": 0.01},
    "SpecialInput": ["x"],
    "InitAtTrueValue": False,
}

posterior_xi_kind = "normal"
posterior_xi_options = {
    "name": "xi",
    "mean": {
    #     "type": "learnable",
    #     "value": 0.0,
    # },
        "type": "neural_net_with_function",
        "input_names": ["y", "x"],
        "hidden_layers": (2000, 2000, 2000),
        "activation_func_name": "SiLU",
        "flagLayerNormalization": True,
    },
    "covariance": {"type": "reduced", "reduced_dim": 10, "std": 0.01},
    "SpecialInput": ["y", "x"],
    "InitAtTrueValue": False,
    "closedFormUpdateAvailable": False,
}

posterior_lmbda_c_kind = "Gamma"  # "Gamma"
posterior_lmbda_c_options = {
    "name": "lmbda_c",
    "gamma": {
        "type": "constant_vector_closed_form",
        "variant": "gaussian",
        "intermediate_key": "r_c",
        "a": 1.0,
        "b": 5e-7,
        "a_0": 5e-7,
        "b_0": 5e-7,
    },
    "SpecialInput": None,
    "expectation_active": True,
    "closedFormUpdateAvailable": True,
}

posterior_jumpPenalty_kind = "Gamma"  # "Gamma"
posterior_jumpPenalty_options = {
    "name": "jumpPenalty",
    "gamma": {
        "type": "constant_vector_closed_form",
        "variant": "gaussian",
        "intermediate_key": "jumps",
        "a": 0.5,
        "b": 1e-6,
        "a_0": 1e-6,
        "b_0": 1e-6,
    },
    "SpecialInput": None,
    "expectation_active": True,
    "closedFormUpdateAvailable": True,
}

# % Tempering steps
num_tempering_steps = 1
num_epochs = [1.75e6]

# %% Optimizer
# general options
num_posterior_samples = 10
default_learning_rate = 3.16227766e-5
default_scheduler_name = None #"MultiStepLR" #(optimizer, , gamma=0.1)  # "StepLR"
default_scheduler_options = {} #"milestones": [3e5, 1e6], "gamma": 0.316227766}  
# {"step_size": 300, "gamma": 0.995}
#     "step_size": 100,  # int, step size for the scheduler
#     "gamma": 0.99,  # float, gamma for the scheduler
# }

# per tempering step options
learning_rate = []  # 1e-3, 1e-3]  # * num_tempering_steps
scheduler_name = []  # "StepLR", "StepLR"]  # "custom_lambda"]
scheduler_options = [
    # {},
    # {"step_size": 500, "gamma": 0.995},
    # {"step_size": 500, "gamma": 0.995},
    # {
    #     "warmup_steps": 10_000,
    #     "base_lr": 1e-6,
    #     "target_lr": 1e-3,
    #     "decay_rate": 0.9999,
    # },
]
# %% Training parameters
max_num_epochs = 3_000_000

when_save_parameters = 50_000
when_log = 1_000
what_to_log = [
    "epoch",
    "elbo",
    "grad_elbo_phi_parts",
    "elbo_parts",
    "mean_square_residual",
    "current_lr",
]

# %% Change problem with iterations
# Posterior
freeze_in_tempering_step = {
    # "lmbda_c": [0]
}  # , "y": [1]}  # "y": [0]} # {"lmbda_c": [0]}
deactivate_posterior_in_tempering_step = {} #"jumpPenalty": [0]}
set_submodule_attrs_posteriors_in_tempering_step = []
lmbda_c_temp_step = 10
update_lambda_c_every_n_iterations = 0.25e5

# Likelihood
deactivate_likelihood_in_tempering_step = {} #"jump": [0]}
set_submodule_attrs_likelihoods_in_tempering_step = [
    # dict(),
    # dict(),
    # dict(),
    # {
    #     "conservation": {
    #         "conservation_covariance": {"std": torch.Tensor([3.15e-3])},
    #     }
    # },
    # {
    #     "conservation": {
    #         "conservation_covariance": {"std": torch.Tensor([1e-3])},
    #     }
    # },
    # {
    #     "conservation": {
    #         "conservation_covariance": {"std": torch.Tensor([5e-4])},
    #     }
    # },
    # {
    #     "conservation": {
    #         "conservation_covariance": {"std": torch.Tensor([2.5e-4])},
    #     }
    # },
    # {
    #     "conservation": {
    #         "conservation_covariance": {"std": torch.Tensor([1e-4])},
    #     }
    # },
]

# Prior
deactivate_prior_in_tempering_step = {"x": [0]}#"jumpPenalty": [0], "jump": [0]}
set_submodule_attrs_priors_in_tempering_step = []


# %% Stages
# Stage 1: train from scratch, constitutive residual ~ Exponential(lmbda_c), one
#          residual per element, lmbda_c frozen.
# Stage 2: start from the stage-1 posterior (lmbda_c is NOT carried over, its
#          shape/model changes), Gaussian constitutive likelihood with one
#          residual per stress component and element, prior on x deactivated.
# These differ in likelihood *type* and in the dimension of lmbda_c, which the
# per-tempering-step dictionaries above cannot express, hence separate stages.
stages = [
    dict(
        name="stage1",
        prior_lmbda_c_options={
            "name": "lmbda_c",
            "gamma": {"type": "constant_scalar", "a": 1.0, "b": 1e-3},
            "SpecialInput": None,
        },
        constitutive_likelihood_kind="exponential",
        constitutive_likelihood_options={
            "name": "constitutive",
            "rate": {"type": "input", "rate_input_key": "lmbda_c"},
            "observation_transformer": {
                "type": "constitutive_residual",
                "OutputName": "r_c",
                "transformerInputKeysStress": "xi",
                "transformerInputKeysConstitutive": ["x", "y"],
            },
            "SpecialInput": ["x", "y", "xi", "lmbda_c"],
        },
        posterior_lmbda_c_options={
            "name": "lmbda_c",
            "gamma": {
                "type": "constant_vector_closed_form",
                "variant": "exponential",
                "intermediate_key": "r_c",
                "a": 1.0,
                "b": 1e-3,
                "a_0": 1.0,
                "b_0": 1e-3,
            },
            "SpecialInput": None,
            "expectation_active": True,
            "closedFormUpdateAvailable": True,
        },
        num_tempering_steps=1,
        num_epochs=[1.75e6],
        default_learning_rate=3.16227766e-4,
        default_scheduler_name="MultiStepLR",
        default_scheduler_options={"milestones": [3e5, 1e6], "gamma": 0.316227766},
        when_log=5_000,
        freeze_in_tempering_step={"lmbda_c": [0]},
        lmbda_c_temp_step=1,
        deactivate_prior_in_tempering_step={},
    ),
    dict(
        name="stage2",
        dont_load=["lmbda_c"],  # applied to the hand-over from stage 1
    ),
]
