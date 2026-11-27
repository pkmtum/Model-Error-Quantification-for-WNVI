# %%
# =============================================================================
# 1. Imports and Global Setup
# =============================================================================

import os
import torch
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from utils.bc_utils import create_boundary_conditions_general
from utils.torch_funcs.function_regular_mesh import regular_2D_mesh
from model.ParameterToField.class_AnalyticalLinearBasisFunctionTriangle import (
    AnalyticalLinearBasisFunction,
)
from model.ParameterToField.class_AnalyticalConstantBasisFunctionTriangle import (
    AnalyticalConstantBasisFunction,
)
from model.ParameterToField.class_AnalyticalConstantBasisFunctionTriangleTensor import (
    AnalyticalConstantBasisFunctionTensor,
)
from model.function_noise import add_noise_dB
from model.ParentClasses.class_Priors_or_Posteriors_Handler import (
    DistributionHandler,
)
from model.class_ELBO import ELBO

# Import PDE and constitutive law classes

from model.theta_loader_util import make_theta_loader
from model.class_SVI import SVI
from utils.class_DynamicLogger import DynamicLogger
from tqdm import tqdm
from utils.function_create_result_dict import create_result_dict
from utils.function_copy_input_file import copy_input_file
from model.postprocessing import plotting_wrapper
from utils.plots.plot_multiple_fields import plot_multiple_fields

# Import from the new utility modules
from utils.initialization_utils import (
    prepare_basis_function_options,
    load_displacement_observations,
    load_stress_reference,
    setup_observation_locations,
    initialize_constitutive_law,
    initialize_pde_harmonic,
    create_stress_calculation_function,
    create_prior,
    prepare_theta_options,
    create_likelihood,
    add_input_dims_to_options_for_neural_net,
    add_input_dims_to_options_for_neural_net_with_function,
    create_posterior,
    load_checkpoint_file,
    apply_checkpoint,
)

from utils.optimization_utils import (
    plotting_before_optimization,
    prepare_elbo_parts_for_logging,
    freeze_and_unfreeze_posterior,
    deactivate_and_activate_distributions,
    set_submodule_attrs_for_distributions,
    mk_folder_for_tempering_step,
    print_new_temp_step_message,
    convergence_criteria_met,
    print_convergence_message,
    criterion_for_saving_parameters_met,
    criterion_for_logging_met,
    print_optimization_finished_message,
    print_logging_message,
    save_parameters,
    print_saving_message,
)

# `inp` (stage settings), `result_path`, `init_state`, `init_u_hat` are injected
# by run_inference.py via runpy.
from utils.torch_funcs.scheduler.scheduler_factory import scheduler_factory

# Set up plotting and device
mpl.rcParams.update(mpl.rcParamsDefault)
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)  # or your cudaIndex
torch.set_default_device(inp.device)
torch.set_default_dtype(inp.default_dtype)



# =============================================================================
# Main Script
# =============================================================================


# 2. Boundary conditions
YToField_options, ThetaToField_options = create_boundary_conditions_general(
    inp.YToField_options_init, inp.ThetaToField_options_init, inp.bc_config, dims=2
)

# 3. Creating fields and grids
XToField_options = prepare_basis_function_options(
    inp.XToField_options_init, inp.scale_x, inp.scale_y
)
YToField_options = prepare_basis_function_options(
    YToField_options, inp.scale_x, inp.scale_y
)
ThetaToField_options = prepare_basis_function_options(
    ThetaToField_options, inp.scale_x, inp.scale_y
)
XiToField_options = prepare_basis_function_options(
    inp.XiToField_options_init, inp.scale_x, inp.scale_y
)

XToField = AnalyticalConstantBasisFunction(XToField_options)
YToField = AnalyticalLinearBasisFunction(YToField_options)
ThetaToField = AnalyticalLinearBasisFunction(ThetaToField_options)
XiToField = AnalyticalConstantBasisFunctionTensor(XiToField_options)

dict_of_dims = {
    "x": int(XToField.num_unknowns),
    "y": int(YToField.num_unknowns),
    "xi": int(XiToField.num_unknowns),
}

center_points = XToField.center_points
nodesMap = XToField.nodesMap
triangulation = mtri.Triangulation(
    XToField_options["bf_grid"][0].flatten().cpu(),
    XToField_options["bf_grid"][1].flatten().cpu(),
    nodesMap.cpu(),
)

# Load brain image and create ground truth material field
img_plain = plt.imread(inp.load_brain_image_path)
brain = img_plain[:, :, 0]
brain_tensor = torch.tensor(brain, dtype=torch.float32)

# Create E_field using the segmentation rules:
# Background: < 0.3 -> E_white
# White matter: > 0.8 -> E_background
# Gray matter: between 0.3 and 0.8 -> E_gray
E_field_tensor = torch.where(
    brain_tensor < 0.3,
    inp.E_white,
    torch.where(
        brain_tensor > 0.8,
        inp.E_background,
        inp.E_gray,  # everything in between (0.3 <= value <= 0.8)
    ),
)
E_field_tensor = torch.flip(E_field_tensor, [0])


def evaluate_field_at_points(E_field_tensor, center_points, scale_x, scale_y):
    """
    Evaluate E_field_tensor at center_points using nearest neighbor interpolation.

    Args:
        E_field_tensor: tensor of shape [H, W] with E values
        center_points: tensor of shape [N, 2] or [2, N] with coordinates in [0, scale_x] x [0, scale_y]
        scale_x, scale_y: domain scales

    Returns:
        E_values: tensor of shape [N] with E values at center_points
    """
    # Ensure center_points is [N, 2]
    if center_points.shape[0] == 2:
        center_points = center_points.T  # Convert [2, N] to [N, 2]

    # Normalize coordinates from [0, scale_x] x [0, scale_y] to [-1, 1] x [-1, 1] for F.grid_sample
    normalized_points = torch.zeros_like(center_points)
    normalized_points[:, 0] = 2 * center_points[:, 0] / scale_x - 1  # x coordinates
    normalized_points[:, 1] = 2 * center_points[:, 1] / scale_y - 1  # y coordinates

    # Reshape for F.grid_sample: [1, 1, N, 2]
    grid = normalized_points.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]

    # Add batch and channel dimensions to E_field_tensor: [1, 1, H, W]
    E_field_batch = E_field_tensor.unsqueeze(0).unsqueeze(0)

    # Sample using nearest neighbor interpolation
    E_values = F.grid_sample(
        E_field_batch,
        grid,
        mode="nearest",
        padding_mode="border",
        align_corners=True,
    )

    # Remove extra dimensions: [1, 1, 1, N] -> [N]
    E_values = E_values.squeeze()

    return E_values


# Evaluate the E field at center points and apply transformation
X_true = evaluate_field_at_points(
    E_field_tensor, center_points, inp.scale_x, inp.scale_y
)
X_groundtruth_1 = torch.log(X_true)  # Apply log transformation
X_groundtruth = torch.log(
    (-X_groundtruth_1 - 1) / (X_groundtruth_1 - 4)
)  # Apply inverse of transformation used in constitutive law

assert torch.allclose(X_true, torch.exp(torch.sigmoid(X_groundtruth) * 5 - 1.0))
print(f"True material field X created with shape: {X_groundtruth.shape}")

# Plot the ground truth material field
plot_multiple_fields(
    X_true.clone().detach(),
    triangulation,
    results_path=result_path,
    titles="True Material Field X",
)

# 4. Create observations
u_true = load_displacement_observations(
    inp.load_u_path, inp.n_obs_s1, inp.n_obs_s2, inp.device, inp.default_dtype
)
u_plot_true = load_displacement_observations(
    inp.load_u_path, inp.general_n_bfs, inp.general_n_bfs, inp.device, inp.default_dtype
)
u_plot_hat = u_plot_true
sigma_true = load_stress_reference(
    inp.load_stress_path,
    XiToField_options["n_bfs_s1"],
    n_bfs_s2=XiToField_options["n_bfs_s2"],
    device=inp.device,
    default_dtype=inp.default_dtype,
    nodesMap=XiToField.nodesMap,
)
u_hat, sigma_u_hat = add_noise_dB(u_true, inp.SNR_in_dB, return_sigma=True)
if init_u_hat is not None:  # keep the noise realisation of the previous stage
    u_hat = init_u_hat.clone().detach()
u_hat_positions = regular_2D_mesh(
    inp.n_obs_s1,
    inp.n_obs_s2,
    on_boundary=True,
    scale_x=inp.scale_x,
    scale_y=inp.scale_y,
)
# u_hat_dict = torch.load(inp.load_noisy_u_path)
# u_hat, sigma_u_hat, u_hat_positions = (
#     u_hat_dict["u_hat"],
#     u_hat_dict["sigma_u_hat"],
#     u_hat_dict["u_hat_positions"],
# )
setup_observation_locations(
    YToField,
    inp.n_obs_s1,
    inp.n_obs_s2,
    u_hat_positions.flatten(start_dim=-2, end_dim=-1).T,
)

# 5. Initialize constitutive law and PDE
constitutive_law = initialize_constitutive_law(
    MaterialModel=inp.MaterialModel,
    XToField=XToField,
    YToField=YToField,
    nu=inp.nu,
    flag_E_plus_one=inp.flag_E_plus_one,
)

# myPDE = initialize_pde(
#     ThetaToField=ThetaToField,
#     XiToField=XiToField,
#     args_rhs=inp.args_rhs,
#     device=inp.device,
# )
myPDE = initialize_pde_harmonic(
    ThetaToField=ThetaToField,
    XiToField=XiToField,
    YToField=YToField,
    args_rhs=inp.args_rhs,
    device=inp.device,
    rho=inp.rho,
    frequency=inp.frequency,
)
# Create stress calculation function for constitutive likelihood
stress_calc_fn = create_stress_calculation_function(constitutive_law)

# 6. Create Priors
priors = DistributionHandler()
priors = create_prior(
    priors,
    inp.prior_x_kind,
    inp.prior_x_options,
    prior_dimension=int(XToField.num_unknowns),
)
priors = create_prior(
    priors,
    inp.prior_y_kind,
    inp.prior_y_options,
    prior_dimension=int(YToField.num_unknowns),
)
priors = create_prior(
    priors,
    inp.prior_xi_kind,
    inp.prior_xi_options,
    prior_dimension=int(XiToField.num_unknowns),
)
priors = create_prior(
    priors,
    inp.prior_lmbda_c_kind,
    inp.prior_lmbda_c_options,
    prior_dimension=int(XToField.num_unknowns),
)
priors = create_prior(
    priors,
    inp.prior_jumpPenalty_kind,
    inp.prior_jumpPenalty_options,
    prior_dimension=int(len(triangulation.neighbors)),
)


def FieldTransformerFunction(**kwargs):
    XField = XToField.eval(**kwargs)
    Intermediate_1 = 5 * torch.sigmoid(XField) - 1.0
    Intermediate_2 = torch.exp(Intermediate_1)
    return Intermediate_2


priors = create_prior(
    priors,
    inp.prior_jump_kind,
    inp.prior_jump_options,
    prior_dimension=int(
        len(triangulation.edges)
        - 2 * (XToField_options["n_bfs_s1"] - 1)
        - 2 * (XToField_options["n_bfs_s2"] - 1)
    ),
    additional_options={
        "observation_transformer": {
            "FieldTransformerFunction": FieldTransformerFunction,
            "FieldNodesMap": nodesMap,
        }
    },
)

# 7. Create Theta Data Set
theta_loader_options = prepare_theta_options(
    theta_options=inp.theta_options,
    BasisFunctionGrid=ThetaToField.bf_grid,
    NumUnknownParameters=ThetaToField.num_unknowns,
    MaskWithGivenValuesForBasisFunctionGrid=ThetaToField.mask_given_values,
)

theta_loader = make_theta_loader(
    builders=theta_loader_options,
    batch_size=inp.theta_batch_size,
)

# 8. Create likelihoods
likelihoods = DistributionHandler()

conservation_likelihood_additional_options = {
    "observation_transformer": {
        "ThetaDataLoader": theta_loader,
        "ResidualFunction": myPDE.forward,
    },
    "mean": {"dim": 1},
    "covariance": {"dim": 1},
}
likelihoods = create_likelihood(
    likelihoods,
    likelihood_kind=inp.conservation_likelihood_kind,
    likelihood_options=inp.conservation_likelihood_options,
    additional_options=conservation_likelihood_additional_options,
)

actual_likelihood_additional_options = {
    "mean": {
        "dim": inp.n_obs_s1 * inp.n_obs_s2 * 2,
        "value": u_hat.clone().detach().T.flatten(),
    },
    "covariance": {"dim": YToField.num_unknowns, "std": sigma_u_hat},
    "observation_transformer": {"FieldTransformerFunction": YToField.eval_at_locations},
}
likelihoods = create_likelihood(
    likelihoods,
    likelihood_kind=inp.actual_likelihood_kind,
    likelihood_options=inp.actual_likelihood_options,
    additional_options=actual_likelihood_additional_options,
)

# Dimension of lmbda_c = number of constitutive residuals: one per stress
# component and element (stage 2 setup, XiToField.n_bfs) or one per element (stage 1).
if inp.constitutive_likelihood_options["observation_transformer"].get(
    "flag_one_residual_per_parameter", False
):
    n_lmbda_c = int(XiToField.n_bfs)
else:
    n_lmbda_c = len(triangulation.triangles)

constitutive_likelihood_additional_options = {
    "mean": {"dim": n_lmbda_c},
    "covariance": {"dim": n_lmbda_c},
    "rate": {"dim": n_lmbda_c},
    "observation_transformer": {
        "ParameterToStress_fn": XiToField.eval,
        "ConstitutiveModel_fn": stress_calc_fn,  # Use the wrapped function
    },
}
likelihoods = create_likelihood(
    likelihoods,
    likelihood_kind=inp.constitutive_likelihood_kind,
    likelihood_options=inp.constitutive_likelihood_options,
    additional_options=constitutive_likelihood_additional_options,
)

# 9. Create Posteriors
posteriors = DistributionHandler()
# check for Initlization at true value
posterior_x_additional_options = (
    {}
    if not inp.posterior_x_options.get("InitAtTrueValue", False)
    else {"mean": {"value": X_groundtruth.clone().detach().flatten()}}
)
posterior_x_options = add_input_dims_to_options_for_neural_net(
    inp.posterior_x_options, dict_of_dims
)
posterior_y_additional_options = (
    {}
    if not inp.posterior_y_options.get("InitAtTrueValue", False)
    else {
        "mean": {"value": u_hat.clone().detach().T.flatten()},
        "covariance": {"std": sigma_u_hat.clone().detach()},
    }
)
posterior_y_options = add_input_dims_to_options_for_neural_net(
    inp.posterior_y_options, dict_of_dims
)
posterior_xi_additional_options = (
    {}
    if not inp.posterior_xi_options.get("InitAtTrueValue", False)
    else {"mean": {"value": sigma_true.clone().detach()[:, [0, 1, 3]].T.flatten()}}
)
posterior_xi_options = add_input_dims_to_options_for_neural_net(
    inp.posterior_xi_options, dict_of_dims
)

posterior_xi_options = add_input_dims_to_options_for_neural_net_with_function(
    posterior_xi_options,
    transformation_fn=stress_calc_fn,
    input_dim=int(XiToField.n_bfs),
)

# Initialize posteriors
posteriors = create_posterior(
    posteriors,
    inp.posterior_x_kind,
    posterior_options=posterior_x_options,
    posterior_dimension=int(XToField.num_unknowns),
    additional_options=posterior_x_additional_options,
)
posteriors = create_posterior(
    posteriors,
    inp.posterior_y_kind,
    posterior_options=posterior_y_options,
    posterior_dimension=int(YToField.num_unknowns),
    additional_options=posterior_y_additional_options,
)
posteriors = create_posterior(
    posteriors,
    inp.posterior_xi_kind,
    posterior_options=posterior_xi_options,
    posterior_dimension=int(XiToField.num_unknowns),
    additional_options=posterior_xi_additional_options,
)
posteriors = create_posterior(
    posteriors,
    inp.posterior_lmbda_c_kind,
    inp.posterior_lmbda_c_options,
    posterior_dimension=n_lmbda_c,
)
posteriors = create_posterior(
    posteriors,
    inp.posterior_jumpPenalty_kind,
    inp.posterior_jumpPenalty_options,
    posterior_dimension=int(
        len(triangulation.edges)
        - 2 * (XToField_options["n_bfs_s1"] - 1)
        - 2 * (XToField_options["n_bfs_s2"] - 1)
    ),
)

# 10. Load checkpoint if available
if init_state is not None:  # in-memory hand-over from the previous stage
    posteriors, u_hat = apply_checkpoint(
        {**init_state, "u_hat": u_hat}, posteriors, dont_load=inp.dont_load, u_hat=u_hat
    )
elif inp.flag_load_checkpoint:
    checkpoint = load_checkpoint_file(inp.checkpoint_path, inp.device)
    posteriors, u_hat = apply_checkpoint(
        checkpoint, posteriors, dont_load=inp.dont_load, u_hat=u_hat
    )

# 11. Initialize ELBO and SVI
default_scheduler_init_fn = scheduler_factory(
    inp.default_scheduler_name, inp.default_scheduler_options
)

elbo = ELBO(
    priors=priors,
    likelihoods=likelihoods,
    posteriors=posteriors,
)

svi = SVI(
    elbo=elbo,
    num_posterior_samples=inp.num_posterior_samples,
    default_scheduler_init_fn=default_scheduler_init_fn,
    default_learning_rate=inp.default_learning_rate,
)

# 12. Sanity check plots
plotting_before_optimization(
    ThetaToField=ThetaToField,
    triangulation=triangulation,
    RightHandSideOptions=inp.args_rhs,
    result_path=result_path,
)

# %% 13. Optimization loop
global_logger = DynamicLogger(inp.what_to_log)
for temp_step in range(inp.num_tempering_steps):
    print_new_temp_step_message(temp_step, inp.num_tempering_steps)
    result_path_tempering = mk_folder_for_tempering_step(temp_step, result_path)
    temp_logger = DynamicLogger(inp.what_to_log)

    # Freeze / Unfreeze posteriors
    posteriors = freeze_and_unfreeze_posterior(
        posteriors, inp.freeze_in_tempering_step, temp_step
    )

    # Deactivate / activate priors, likelihoods & posteriors
    priors = deactivate_and_activate_distributions(
        priors, inp.deactivate_prior_in_tempering_step, temp_step
    )
    likelihoods = deactivate_and_activate_distributions(
        likelihoods, inp.deactivate_likelihood_in_tempering_step, temp_step
    )
    posteriors = deactivate_and_activate_distributions(
        posteriors, inp.deactivate_posterior_in_tempering_step, temp_step
    )

    # Set submodule attributes for each handler if provided
    set_submodule_attrs_for_distributions(
        posteriors, inp.set_submodule_attrs_posteriors_in_tempering_step, temp_step
    )
    set_submodule_attrs_for_distributions(
        priors, inp.set_submodule_attrs_priors_in_tempering_step, temp_step
    )
    set_submodule_attrs_for_distributions(
        likelihoods,
        inp.set_submodule_attrs_likelihoods_in_tempering_step,
        temp_step,
    )

    # scheduler_init_fn = scheduler_factory(
    #     inp.scheduler_name[temp_step], inp.scheduler_options[temp_step]
    # )

    for epoch in tqdm(range(inp.max_num_epochs)):
        if temp_step > (inp.lmbda_c_temp_step - 1):
            if epoch % inp.update_lambda_c_every_n_iterations == 0:
                # Freeze / Unfreeze posteriors
                posteriors = freeze_and_unfreeze_posterior(posteriors, {}, 0)
                # svi.reset()
            elif epoch % inp.update_lambda_c_every_n_iterations == 1:
                # Freeze / Unfreeze posteriors
                posteriors = freeze_and_unfreeze_posterior(
                    posteriors, {"lmbda_c": [0, 1, 2]}, temp_step
                )

        (
            elbo,
            grad_elbo_phi_parts,
            samples,
            intermediates,
            E_log_likelihoods,
            E_log_priors,
            E_entropy_posteriors,
        ) = svi.step()
        elbo = elbo.clone().detach()

        mean_square_residual = torch.sum(
            torch.mean(intermediates["r"] ** 2, dim=0), dim=0
        )

        if criterion_for_saving_parameters_met(epoch, inp.when_save_parameters):
            # Get current learning rate from optimizer
            current_lr = svi.optimizer.param_groups[0]["lr"]

            # Neeed for logging
            elbo_parts = prepare_elbo_parts_for_logging(
                [E_log_likelihoods, E_log_priors, E_entropy_posteriors],
                name_prefixes=["E_log_prob_", "E_log_prob_", "E_entropy_"],
            )
            samples = prepare_elbo_parts_for_logging([samples], name_prefixes=[""])

            # create a new directory for the current iteration.
            result_path_tempering_itermediate = mk_folder_for_tempering_step(
                epoch, result_path_tempering
            )

            print_logging_message(
                current_iteration=epoch,
                mean_square_residual=mean_square_residual,
                elbo=elbo,
            )
            global_logger.log(locals())
            temp_logger.log(locals())

            print_saving_message()
            save_parameters(posteriors, result_path_tempering_itermediate, epoch)
            temp_logger.save(result_path_tempering_itermediate)

            plotting_wrapper(temp_logger, result_path_tempering_itermediate, locals())

            if convergence_criteria_met(epoch, int(inp.num_epochs[temp_step])):
                print_convergence_message(epoch, int(inp.num_epochs[temp_step]))
                break

        elif criterion_for_logging_met(epoch, inp.when_log):
            # Get current learning rate from optimizer
            current_lr = svi.optimizer.param_groups[0]["lr"]

            print_logging_message(
                current_iteration=epoch,
                mean_square_residual=mean_square_residual,
                elbo=elbo,
            )
            # Neeed for logging
            elbo_parts = prepare_elbo_parts_for_logging(
                [E_log_likelihoods, E_log_priors, E_entropy_posteriors],
                name_prefixes=["E_log_prob_", "E_log_prob_", "E_entropy_"],
            )
            samples = prepare_elbo_parts_for_logging([samples], name_prefixes=[""])
            global_logger.log(locals())
            temp_logger.log(locals())

print_optimization_finished_message()
global_logger.save(result_path)
plotting_wrapper(global_logger, result_path, locals())

final_state = posteriors.state_dict()
final_u_hat = u_hat.clone().detach()
