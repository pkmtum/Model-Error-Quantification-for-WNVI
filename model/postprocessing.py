import torch
import numpy as np
import warnings
import matplotlib.tri as mtri

# plots
from utils.function_count_triangulation_fields_in_vector import (
    count_triangulation_fields_in_vector,
)
from utils.plots.plot_ELBO import plot_ELBO_parts
from utils.plots.plot_parameter_evolution import plot_parameter_evolution
from utils.plots.plot_multiple_fields import plot_multiple_fields
from utils.plots.plot_line_w_running_average import plot_line_w_runnning_avg
from utils.plots.plot_true_mean_std_binary import plot_true_mean_std_binary
from utils.plots.plot_dict_entries_norm import plot_dict_entries_norm
from utils.plots.plot_abs_error_std import plot_abs_error_std

from model.ParentClasses.class_Priors_or_Posteriors_Handler import DistributionHandler
from model.ParentClasses.class_ParameterToField import ParameterToField
from utils.class_DynamicLogger import DynamicLogger
from model.ParentClasses.class_BaseConstitutiveLaw import BaseConstitutiveLaw
import os
from glob import glob

PLOT_IGNORE = ["bias", "weight"]


def plotting_wrapper(logger: DynamicLogger, result_path: str, vars: dict) -> None:
    """
    A wrapper function to plot the variables from the dynamic logger.
    This is a placeholder implementation and should be customized as needed.
    """
    my_log = logger
    posterior = vars["posteriors"]
    # myPDE = vars["myPDE"]
    myConstitutiveLaw = vars["constitutive_law"]
    ParameterToFieldDict = {}
    ParameterToFieldDict["x"] = vars["XToField"]
    ParameterToFieldDict["y"] = vars["YToField"]
    ParameterToFieldDict["theta"] = vars["ThetaToField"]
    ParameterToFieldDict["xi"] = vars["XiToField"]
    triangulation = vars["triangulation"]
    MF_true = vars["X_groundtruth"]
    u_true = vars["u_plot_true"]
    u_hat_full = vars["u_plot_hat"]
    sigma_u_hat = vars["sigma_u_hat"]
    sigma_true = vars["sigma_true"]
    result_path = result_path

    postprocessing(
        my_log,
        myConstitutiveLaw,
        posterior,
        ParameterToFieldDict,
        triangulation,
        MF_true,
        u_true,
        u_hat_full,
        sigma_u_hat,
        sigma_true,
        result_path,
    )


def safe_plot(plot_func, *args, **kwargs):
    try:
        plot_func(*args, **kwargs)
    except Exception as e:
        warnings.warn(f"Plotting failed in {plot_func.__name__}: {e}")


def plot_evolutions_from_loaded_data(
    loading_path: str, saving_path: str, plot_ignore: list[str]
) -> None:
    """
    Loads posterior parameter data from state_dict_*.pt files in each subdirectory of loading_path,
    and plots the evolution of each parameter key over the iterations. For high-dimensional vectors,
    only up to 100 evenly spaced entries are plotted.
    Skips plotting for parameter names containing any string in plot_ignore.
    """

    # Find all subdirectories in loading_path
    subdirs = [
        os.path.join(loading_path, d)
        for d in os.listdir(loading_path)
        if os.path.isdir(os.path.join(loading_path, d))
    ]

    # Sort subdirectories by the numeric value at the end of their names
    def extract_number_from_dirname(path):
        dirname = os.path.basename(path)
        # Split by underscore and take the last part, then convert to int
        try:
            return int(dirname.split("_")[-1])
        except ValueError:
            # If conversion fails, return a very large number to put it at the end
            return float("inf")

    subdirs.sort(key=extract_number_from_dirname)

    param_evolution = {}
    for subdir in subdirs:
        # Find all state_dict_*.pt files and sort by iteration number
        state_files = sorted(
            glob(os.path.join(subdir, "state_dict_*.pt")),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]),
        )
        if not state_files:
            continue
        # Load all posterior_parameters for each iteration
        for state_file in state_files:
            state = torch.load(state_file, map_location="cpu")
            # Flatten nested OrderedDicts: {'x': OrderedDict([('name', tensor), ...]), ...}
            posterior_parameters = {}
            for group_key, group_val in state.items():
                if isinstance(group_val, dict):
                    for param_key, param_val in group_val.items():
                        if any(ignore_str in param_key for ignore_str in plot_ignore):
                            continue
                        posterior_parameters[param_key] = param_val
                else:
                    posterior_parameters[group_key] = group_val
            for key, value in posterior_parameters.items():
                # Convert to numpy array if tensor
                if isinstance(value, torch.Tensor):
                    # Flatten multi-dimensional tensors to 1D vectors
                    arr = value.flatten().cpu().numpy()
                elif (
                    isinstance(value, (list, tuple))
                    and len(value) > 0
                    and isinstance(value[0], torch.Tensor)
                ):
                    # Stack tensors and flatten if multi-dimensional
                    stacked = torch.stack(value)
                    arr = stacked.flatten().cpu().numpy()
                else:
                    arr = np.array(value).flatten()
                if key not in param_evolution:
                    param_evolution[key] = []
                param_evolution[key].append(arr)

    # Calculate gamma expectations (E = a/b) for gamma parameters
    gamma_expectations = {}
    for key in param_evolution.keys():
        if key.endswith("_gamma.a"):
            # Find corresponding b parameter
            base_name = key[:-8]  # Remove '_gamma.a'
            b_key = base_name + "_gamma.b"
            if b_key in param_evolution:
                # Calculate expectation E = a/b for each iteration
                a_values = np.array(param_evolution[key])
                b_values = np.array(param_evolution[b_key])
                expectation_values = a_values / b_values
                expectation_key = base_name + "_gamma.expectation"
                gamma_expectations[expectation_key] = expectation_values

    # Add gamma expectations to param_evolution
    param_evolution.update(gamma_expectations)

    # Now plot the evolution for each parameter key
    for key, values in param_evolution.items():
        # Skip plotting if key contains any string in plot_ignore
        if any(ignore_str in key for ignore_str in plot_ignore):
            continue
        values = np.array(values)
        # If values is more than 1D and has more than 100 entries in the last dimension, subsample
        if values.ndim > 1 and values.shape[-1] > 100:
            idxs = np.linspace(0, values.shape[-1] - 1, 100, dtype=int)
            values = values[..., idxs]
        # Reshape to (iterations, features)
        values_flat = values.reshape(values.shape[0], -1)

        plot_parameter_evolution(
            values_flat.T, results_path=saving_path, param_name=key, plt_show=False
        )


def postprocessing(
    my_log: DynamicLogger,
    myConstitutiveLaw: BaseConstitutiveLaw,
    posterior: DistributionHandler,
    ParameterToFieldDict: dict[str, ParameterToField],
    triangulation: mtri.Triangulation,
    MF_true: torch.Tensor,
    u_true: torch.Tensor,
    u_hat_full: torch.Tensor,
    sigma_u_hat: torch.Tensor,
    sigma_true: torch.Tensor,
    result_path: str,
) -> None:
    # Plot ELBO and gradients
    plot_elbo_and_gradients(my_log, result_path)

    # Plot parameter evolution
    plot_evolutions_from_loaded_data(
        loading_path=os.path.dirname(result_path),
        saving_path=result_path,
        plot_ignore=PLOT_IGNORE,
    )

    samples = posterior.sample(num_samples=1_000)

    # Material field
    if "x" in samples:
        plot_material_field(
            samples, ParameterToFieldDict, MF_true, triangulation, result_path
        )
    else:
        warnings.warn("Plotting: No x in the samples")

    # Displacement field
    if "y" in samples:
        plot_displacement_field(
            samples,
            ParameterToFieldDict,
            u_true,
            u_hat_full,
            sigma_u_hat,
            triangulation,
            result_path,
        )
    else:
        warnings.warn("Plotting: No 'y' in the samples for plotting.")

    # Stress field
    if "xi" in samples:
        plot_stress_field(
            samples,
            ParameterToFieldDict,
            sigma_true,
            myConstitutiveLaw,
            triangulation,
            result_path,
        )
    else:
        warnings.warn("Plotting: No 'xi' in the samples for plotting.")

    # Lambda_c field
    if "lmbda_c" in samples:
        plot_lambda_c_field(samples, posterior, triangulation, result_path)

    # Mean square residual
    safe_plot(
        plot_line_w_runnning_avg,
        [np.abs(my_log.history["mean_square_residual"])],
        [r"E[$r^2$]"],
        running_avg_range=None,
        results_path=result_path,
    )

    # # Theta field
    # safe_plot(
    #     plot_multiple_fields,
    #     ParameterToFieldDict["theta"].eval(torch.tensor(my_log.theta[-1])),
    #     triangulation,
    #     results_path=result_path,
    #     titles="theta",
    # )


def plot_material_field(
    samples, ParameterToFieldDict, MF_true, triangulation, result_path
):
    try:
        x_sample = samples["x"].clone().detach()
        x_sample = torch.sigmoid(x_sample) * 5 - 1.0
        log_MF_true = torch.sigmoid(MF_true) * 5 - 1.0
        MF_samples = ParameterToFieldDict["x"].eval(x_sample)
        MF_mean = MF_samples.mean(dim=0)
        MF_std = torch.std(MF_samples, dim=0)
        safe_plot(
            plot_true_mean_std_binary,
            log_MF_true,
            MF_mean,
            MF_std,
            triangulation,
            suptitle=r"Posterior - log Material Field $log(E)$",
            results_path=result_path,
        )

        x_sample2 = torch.exp(x_sample)
        MF_samples2 = ParameterToFieldDict["x"].eval(x_sample2)
        MF_mean2 = MF_samples2.mean(dim=0)
        MF_std2 = torch.std(MF_samples2, dim=0)
        safe_plot(
            plot_true_mean_std_binary,
            torch.exp(log_MF_true),
            MF_mean2,
            MF_std2,
            triangulation,
            suptitle=r"Posterior - Material Field $E$",
            results_path=result_path,
        )

        safe_plot(
            plot_multiple_fields,
            torch.log(MF_std),
            triangulation,
            results_path=result_path,
            titles="log(MF_std) of log(E)",
        )
    except Exception as e:
        warnings.warn(f"Material field plotting failed: {e}")


def plot_displacement_field(
    samples,
    ParameterToFieldDict,
    u_true,
    u_hat_full,
    sigma_u_hat,
    triangulation,
    result_path,
):
    try:
        y_sample = samples["y"].clone().detach()
        u_samples1 = ParameterToFieldDict["y"].apply_know_values(y_sample)
        u_samples = (
            torch.unflatten(
                u_samples1,
                -1,
                (
                    2,
                    ParameterToFieldDict["y"].bf_grid.size(dim=-2),
                    ParameterToFieldDict["y"].bf_grid.size(dim=-1),
                ),
            )
            .flatten(start_dim=-2)
            .transpose(dim0=-2, dim1=-1)
        )
        u_mean = u_samples.mean(dim=0)
        u_std = torch.std(u_samples, dim=0)
        safe_plot(
            plot_true_mean_std_binary,
            u_true[:, 0],
            u_mean[:, 0],
            u_std[:, 0],
            triangulation,
            suptitle=r"Posterior - Displacements $u_1$",
            results_path=result_path,
        )
        safe_plot(
            plot_true_mean_std_binary,
            u_true[:, 1],
            u_mean[:, 1],
            u_std[:, 1],
            triangulation,
            suptitle=r"Posterior - Displacements $u_2$",
            results_path=result_path,
        )
        safe_plot(
            plot_abs_error_std,
            u_hat_full[:, 0],
            u_mean[:, 0],
            sigma_u_hat,
            triangulation,
            suptitle=r"Absolute Error $u_1$",
            results_path=result_path,
        )
        safe_plot(
            plot_abs_error_std,
            u_hat_full[:, 1],
            u_mean[:, 1],
            sigma_u_hat,
            triangulation,
            suptitle=r"Absolute Error $u_2$",
            results_path=result_path,
        )
        # u_part_obs = u_hat_full

        # safe_plot(
        #     plot_line_with_std,
        #     u_mean[1, :, 0],
        #     u_std[1, :, 0],
        #     true_value=u_true[1, :, 0],
        #     obs_points=u_part_obs[1, :, 0],
        #     title=r"Solution field $u_2$ at $x_1=0$",
        #     results_path=result_path,
        # )
        # safe_plot(
        #     plot_line_with_std,
        #     u_mean[0, 0, :],
        #     u_std[0, 0, :],
        #     true_value=u_true[0, 0, :],
        #     obs_points=u_part_obs[0, 0, :],
        #     title=r"Solution field $u_2$ at $x_2=0$",
        #     results_path=result_path,
        # )

        du_samples = ParameterToFieldDict["y"].eval_grad_s(y_sample)
        du_mean = du_samples.mean(dim=0)
        du_mean = du_mean.flatten(start_dim=-2)
        safe_plot(
            plot_multiple_fields,
            du_mean,
            triangulation,
            results_path=result_path,
            titles=["dux_dx", "dux_dy", "duy_dx", "duy_dy"],
        )

        du_samples = ParameterToFieldDict["y"].eval_grad_s(y_sample)
        eps = 0.5 * (torch.einsum("...ijk->...ikj", du_samples) + du_samples)
        eps_mean = eps.mean(dim=0)
        eps_mean = eps_mean.flatten(start_dim=-2)
        eps_std = torch.std(eps, dim=0)
        eps_std = eps_std.flatten(start_dim=-2)
        safe_plot(
            plot_multiple_fields,
            torch.stack(
                (
                    eps_mean[:, 0],
                    eps_mean[:, 1],
                    eps_mean[:, 3],
                    eps_std[:, 0],
                    eps_std[:, 1],
                    eps_std[:, 3],
                ),
                dim=0,
            ).T,
            triangulation,
            results_path=result_path,
            titles=[
                "eps_11",
                "eps_12",
                "eps_22",
                "eps_std_11",
                "eps_std_12",
                "eps_std_22",
            ],
        )
    except Exception as e:
        warnings.warn(f"Displacement field plotting failed: {e}")


def plot_stress_field(
    samples,
    ParameterToFieldDict,
    sigma_true,
    myConstitutiveLaw,
    triangulation,
    result_path,
):
    try:
        xi_sample = samples["xi"].clone().detach()
        xi_mean = torch.mean(xi_sample, dim=0)
        xi_std = torch.std(xi_sample, dim=0)
        len_xi = len(xi_mean)
        xi_mean_11, xi_mean_12, xi_mean_22 = (
            xi_mean[: len_xi // 3],
            xi_mean[len_xi // 3 : 2 * len_xi // 3],
            xi_mean[2 * len_xi // 3 :],
        )
        xi_std_11, xi_std_12, xi_std_22 = (
            xi_std[: len_xi // 3],
            xi_std[len_xi // 3 : 2 * len_xi // 3],
            xi_std[2 * len_xi // 3 :],
        )
        sigma_11 = xi_mean_11
        sigma_12 = xi_mean_12
        sigma_22 = xi_mean_22
        sigma_std_11 = xi_std_11
        sigma_std_12 = xi_std_12
        sigma_std_22 = xi_std_22
        stacked_sigma = torch.stack(
            (
                sigma_11,
                sigma_12,
                sigma_22,
                sigma_std_11,
                sigma_std_12,
                sigma_std_22,
            ),
            dim=0,
        ).T

        safe_plot(
            plot_multiple_fields,
            stacked_sigma,
            triangulation,
            results_path=result_path,
            titles=[
                "sigma_11",
                "sigma_12",
                "sigma_22",
                "sigma_std_11",
                "sigma_std_12",
                "sigma_std_22",
            ],
        )

        sigma_calc = (
            myConstitutiveLaw.calculate_stress(x=samples["x"], y=samples["y"])
            .clone()
            .detach()
        )
        sigma_calc_mean = torch.mean(sigma_calc, dim=0)
        sigma_calc_std = torch.std(sigma_calc, dim=0)
        sigma_calc_11 = sigma_calc_mean[:, 0, 0]
        sigma_calc_12 = sigma_calc_mean[:, 1, 0]
        sigma_calc_22 = sigma_calc_mean[:, 1, 1]
        sigma_calc_std_11 = sigma_calc_std[:, 0, 0]
        sigma_calc_std_12 = sigma_calc_std[:, 1, 0]
        sigma_calc_std_22 = sigma_calc_std[:, 1, 1]
        stacked_sigma_calc = torch.stack(
            (
                sigma_calc_11,
                sigma_calc_12,
                sigma_calc_22,
                sigma_calc_std_11,
                sigma_calc_std_12,
                sigma_calc_std_22,
            ),
            dim=0,
        ).T
        safe_plot(
            plot_multiple_fields,
            stacked_sigma_calc,
            triangulation,
            results_path=result_path,
            titles=[
                "sigma_calc_11",
                "sigma_calc_12",
                "sigma_calc_22",
                "sigma_calc_std_11",
                "sigma_calc_std_12",
                "sigma_calc_std_22",
            ],
        )
        safe_plot(
            plot_true_mean_std_binary,
            sigma_11,
            sigma_calc_11,
            sigma_calc_std_11,
            triangulation,
            suptitle=r"Error in $sigma_{11}$",
            results_path=result_path,
        )
        safe_plot(
            plot_true_mean_std_binary,
            sigma_12,
            sigma_calc_12,
            sigma_calc_std_12,
            triangulation,
            suptitle=r"Error in $sigma_{12}$",
            results_path=result_path,
        )
        safe_plot(
            plot_true_mean_std_binary,
            sigma_22,
            sigma_calc_22,
            sigma_calc_std_22,
            triangulation,
            suptitle=r"Error in $sigma_{22}$",
            results_path=result_path,
        )

        dsigma_11 = sigma_11 - sigma_true[:, 0]
        dsigma_12 = sigma_12 - sigma_true[:, 1]
        dsigma_22 = sigma_22 - sigma_true[:, -1]
        dsigma_calc_11 = sigma_calc_11 - sigma_true[:, 0]
        dsigma_calc_12 = sigma_calc_12 - sigma_true[:, 1]
        dsigma_calc_22 = sigma_calc_22 - sigma_true[:, -1]
        sigma_true_short = sigma_true[:, [0, 1, -1]]

        plot_helper_sigma1 = torch.concat(
            (
                sigma_true_short,
                sigma_11.unsqueeze(-1),
                sigma_12.unsqueeze(-1),
                sigma_22.unsqueeze(-1),
                dsigma_11.unsqueeze(-1),
                dsigma_12.unsqueeze(-1),
                dsigma_22.unsqueeze(-1),
            ),
            dim=-1,
        )
        plot_helper_sigma2 = torch.concat(
            (
                sigma_true_short,
                sigma_calc_11.unsqueeze(-1),
                sigma_calc_12.unsqueeze(-1),
                sigma_calc_22.unsqueeze(-1),
                dsigma_calc_11.unsqueeze(-1),
                dsigma_calc_12.unsqueeze(-1),
                dsigma_calc_22.unsqueeze(-1),
            ),
            dim=-1,
        )
        safe_plot(
            plot_multiple_fields,
            plot_helper_sigma1,
            triangulation,
            results_path=result_path,
            titles=[
                "sigma_true_11",
                "sigma_true_12",
                "sigma_true_22",
                "sigma_PDE_11",
                "sigma_PDE_12",
                "sigma_PDE_22",
                "dsigma_11",
                "dsigma_12",
                "dsigma_22",
            ],
            norm=True,
        )
        safe_plot(
            plot_multiple_fields,
            plot_helper_sigma2,
            triangulation,
            results_path=result_path,
            titles=[
                "sigma_true_11",
                "sigma_true_12",
                "sigma_true_22",
                "sigma_calc_11",
                "sigma_calc_12",
                "sigma_calc_22",
                "dsigma_calc_11",
                "dsigma_calc_12",
                "dsigma_calc_22",
            ],
            norm=True,
        )

        r_c_squared = (
            (sigma_calc[:, :, 0, 0] - xi_mean_11) ** 2
            + 2 * (sigma_calc[:, :, 1, 0] - xi_mean_12) ** 2
            + (sigma_calc[:, :, 1, 1] - xi_mean_22) ** 2
        )
        r_c = torch.sqrt(r_c_squared)
        r_c_mean = r_c.mean(dim=0)
        r_c_std = r_c.std(dim=0)
        r_c_mean_field = r_c_mean
        r_c_std_field = r_c_std
        safe_plot(
            plot_true_mean_std_binary,
            torch.zeros_like(r_c_mean_field),
            r_c_mean_field,
            r_c_std_field,
            triangulation,
            suptitle=r"Error in $r_c$",
            results_path=result_path,
        )

    except Exception as e:
        warnings.warn(f"Stress field plotting failed: {e}")


def plot_lambda_c_field(samples, posterior, triangulation, result_path):
    try:
        lmbda_c = posterior.Distributions["lmbda_c"].mean().clone().detach()
    except Exception:
        if samples["lmbda_c"].dim() > 1:
            lmbda_c = samples["lmbda_c"].mean(dim=0).clone().detach()
        else:
            lmbda_c = samples["lmbda_c"].clone().detach()

    if lmbda_c.size() == torch.Size([]):
        warnings.warn("Plotting: lmbda_c is scalar. We cant plot those.")
        return

    try:
        num_fields, num_nodes, field_type = count_triangulation_fields_in_vector(
            lmbda_c, triangulation
        )
        lmbda_c = lmbda_c.reshape(-1, num_nodes, num_fields).transpose(dim0=-1, dim1=-2)
    except:
        pass

    safe_plot(
        plot_multiple_fields,
        lmbda_c,
        triangulation,
        results_path=result_path,
        titles="lambda_c",
    )
    safe_plot(
        plot_multiple_fields,
        torch.log(lmbda_c),
        triangulation,
        results_path=result_path,
        titles="ln_lambda_c",
    )
    # safe_plot(
    #     plot_multiple_fields,
    #     lmbda_c * r_c_mean_field**2,
    #     triangulation,
    #     results_path=result_path,
    #     titles="lambda_c * r_c**2",
    # )


def plot_elbo_and_gradients(my_log, result_path):
    grad_elbo = [
        sum(values) for values in zip(*my_log.history["grad_elbo_phi_parts"].values())
    ]
    safe_plot(
        plot_line_w_runnning_avg, grad_elbo, "grad_elbo", results_path=result_path
    )
    safe_plot(
        plot_ELBO_parts,
        my_log.history["elbo_parts"],
        my_log.history["epoch"],
        result_path,
    )
    elbo_plot = my_log.history["elbo"]
    const = 0.0
    if len(elbo_plot) > 2:
        elbo_plot = np.asarray(elbo_plot)
        elbo_plot[:2] = np.nan
        const = elbo_plot[2]
        elbo_plot = elbo_plot - const
    safe_plot(
        plot_line_w_runnning_avg,
        [elbo_plot],
        [f"ELBO + {const:.2e}"],
        running_avg_range=None,
        results_path=result_path,
        log_scale=False,
    )
    safe_plot(
        plot_dict_entries_norm,
        my_log.history["grad_elbo_phi_parts"],
        my_log.history["epoch"],
        title="grad_elbo_phi",
        results_path=result_path,
    )

    # Plot learning rate if available
    if "current_lr" in my_log.history:
        safe_plot(
            plot_line_w_runnning_avg,
            [my_log.history["current_lr"]],
            ["Learning Rate"],
            running_avg_range=None,
            results_path=result_path,
            log_scale=False,
        )
