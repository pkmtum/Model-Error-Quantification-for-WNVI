import os
import torch
import matplotlib.tri as mtri
import psutil
from mergedeep import merge

from model.ParentClasses.class_Priors_or_Posteriors_Handler import (
    DistributionHandler,
)
from utils.plots.plot_BoundaryConditions import plot_Neumann_BCs, plot_Dirichlet_BCs


def plotting_before_optimization(
    ThetaToField,
    triangulation: mtri.Triangulation,
    RightHandSideOptions: dict,
    result_path: str,
) -> None:
    plot_Neumann_BCs(
        RightHandSideOptions=RightHandSideOptions,
        ThetaToField=ThetaToField,
        triangulation=triangulation,
        result_path=result_path,
    )
    plot_Dirichlet_BCs(
        ThetaToField=ThetaToField, triangulation=triangulation, result_path=result_path
    )


def prepare_elbo_parts_for_logging(
    list_of_elbo_parts: list[dict[str, torch.Tensor]], name_prefixes: list[str] = [""]
) -> dict[str, torch.Tensor]:
    """Prepare ELBO parts for logging by flattening the list of dictionaries into a single dictionary.

    Args:
        list_of_elbo_parts (list[dict[str, torch.Tensor]]): A list of dictionaries containing ELBO parts.
        name_prefixes (list[str], optional): A list of prefixes to prepend to the keys in the ELBO parts dictionaries. Defaults to [""].

    Returns:
        dict[str, torch.Tensor]: A single dictionary containing all ELBO parts with their keys {prefix}+{name}.
    """
    elbo_parts = dict()
    if name_prefixes == [""]:
        name_prefixes = [""] * len(list_of_elbo_parts)
    else:
        assert len(name_prefixes) == len(list_of_elbo_parts), (
            "Number of name prefixes must match the number of ELBO parts dictionaries."
        )

    for elbo_part, name_prefix in zip(list_of_elbo_parts, name_prefixes):
        elbo_parts = collect_dict_entries(elbo_parts, elbo_part, name_prefix)

    return elbo_parts


def collect_dict_entries(
    target_dict: dict[str, torch.Tensor],
    sample_dict: dict[str, torch.Tensor],
    key_prefix: str,
) -> dict[str, torch.Tensor]:
    """Collect ELBO parts, clone and detach them to avoid in-place operations.

    Args:
        target_dict (dict[str, torch.Tensor]): Dictionary of ELBO parts.
        sample_dict (dict[str, torch.Tensor]): Dictionary of ELBO entries.

    Returns:
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: A tuple containing the collected ELBO parts and samples.
    """
    for name, sample in sample_dict.items():
        target_dict[key_prefix + name] = sample.clone().detach()
    return target_dict


def freeze_and_unfreeze_posterior(
    posterior: DistributionHandler,
    freeze_in_iteration: dict,
    current_iteration: int,
) -> DistributionHandler:
    for key in posterior.Distributions.keys():
        # ... and check if it as information on beeing frozen in the current iteration
        if key in freeze_in_iteration:
            if current_iteration in freeze_in_iteration[key]:
                posterior.freeze_parameters(key)
                print("Freezing {} in iteration {}".format(key, current_iteration))
            else:
                posterior.unfreeze_parameters(key)
                print("Unfreezing {} in iteration {}".format(key, current_iteration))
        else:
            posterior.unfreeze_parameters(key)
            print("Unfreezing {} in iteration {}".format(key, current_iteration))

    return posterior


def deactivate_and_activate_distributions(
    DistributionHandler: DistributionHandler,
    deactivate_in_iteration: dict,
    current_iteration: int,
) -> DistributionHandler:
    for key in DistributionHandler.Distributions.keys():
        # ... and check if it as information on beeing deactivated in the current iteration
        if key in deactivate_in_iteration:
            if current_iteration in deactivate_in_iteration[key]:
                DistributionHandler.deactivate(key)
                print("Deactivating {} in iteration {}".format(key, current_iteration))
            else:
                DistributionHandler.activate(key)
                print("Activating {} in iteration {}".format(key, current_iteration))
        else:
            DistributionHandler.activate(key)
            print("Activating {} in iteration {}".format(key, current_iteration))

    return DistributionHandler


def set_submodule_attrs_for_distributions(
    DistributionHandler: DistributionHandler,
    attrs_per_tempering_step: list[dict],
    current_iteration: int,
) -> None:
    """
    Sets submodule attributes for a distribution handler for the current tempering step.
    Skips if the list is empty or the dict for the current step is empty.
    """
    if not attrs_per_tempering_step:
        return
    if current_iteration >= len(attrs_per_tempering_step):
        return
    attrs_dict = attrs_per_tempering_step[current_iteration]
    if not attrs_dict:
        return
    DistributionHandler.set_submodule_attrs(attrs_dict)


def mk_folder_for_tempering_step(
    current_iteration: int,
    absolute_result_path: str,
    suffix_folder_name: str = "",
) -> str:
    # create a new directory for the current temperature
    folder = (
        f"temp_{current_iteration}_{suffix_folder_name}"
        if suffix_folder_name
        else f"temp_{current_iteration}"
    )
    result_path_temp = os.path.join(absolute_result_path, folder)
    os.makedirs(result_path_temp, exist_ok=True)
    return result_path_temp


def print_new_temp_step_message(
    current_iteration: int,
    maximum_iterations: int,
) -> None:
    print("+++++++++++++++++++++++++++++++++++++++++++++++")
    print(
        f"New tempering step: Iteration {int(current_iteration) + 1} / {int(maximum_iterations)}"
    )
    print("+++++++++++++++++++++++++++++++++++++++++++++++")


def convergence_criteria_met(
    current_iteration: int, maximum_iterations_in_current_temp_step: int
):
    return current_iteration >= maximum_iterations_in_current_temp_step


def print_convergence_message(
    current_iteration: int, maximum_iterations_in_current_temp_step: int
) -> None:
    print("+++++++++++++++++++++++++++++++++++++++++++++++")
    print(
        f"Convergence criteria met in iteration {int(current_iteration) + 1} / {int(maximum_iterations_in_current_temp_step)}"
    )
    print("+++++++++++++++++++++++++++++++++++++++++++++++")


def criterion_for_saving_parameters_met(
    current_iteration: int,
    save_every_n_iterations: int,
) -> bool:
    return (current_iteration) % save_every_n_iterations == 0


def criterion_for_logging_met(
    current_iteration: int,
    log_every_n_iterations: int,
) -> bool:
    return (current_iteration) % log_every_n_iterations == 0


def print_optimization_finished_message() -> None:
    print("+++++++++++++++++++++++++++++++++++++++++++++++")
    print("Optimization finished.")
    print("+++++++++++++++++++++++++++++++++++++++++++++++")


def print_logging_message(
    current_iteration: int,
    mean_square_residual: torch.Tensor,
    elbo: torch.Tensor,
) -> None:
    print("Iteration: {}".format(current_iteration))
    print("ELBO:   ({:.2E})".format(elbo))
    print("E[res^2]:            {:.2E}".format(mean_square_residual))
    print(
        "Memory usage: {:.2f} GB".format(
            psutil.Process(os.getpid()).memory_info().rss / 1024**3
        )
    )


def save_parameters(
    posterior: DistributionHandler,
    absolute_result_path: str,
    current_iteration: int,
    additional_quantities: dict = {},
):
    state_dict = posterior.state_dict()

    state_dict_appended = merge(state_dict, additional_quantities)

    # save first, so I still have the data, even when the postprocessing crashes
    torch.save(
        state_dict_appended,
        absolute_result_path + f"/state_dict_{current_iteration}.pt",
    )


def print_saving_message() -> None:
    print("Saving parameters...")  # Print a message to indicate saving is in progress
