import os
import torch
from typing import Callable, Tuple
import warnings
from mergedeep import merge

from model.PDEs.HarmonicBalanceOfLinearMomentum import HarmonicBalanceOfLinearMomentum
from model.ParameterToField.class_AnalyticalConstantBasisFunctionTriangleTensor import (
    AnalyticalConstantBasisFunctionTensor,
)
from utils.torch_funcs.function_regular_mesh import regular_2D_mesh
from model.ParameterToField.class_AnalyticalLinearBasisFunctionTriangle import (
    AnalyticalLinearBasisFunction,
)
from model.ParameterToField.class_AnalyticalConstantBasisFunctionTriangle import (
    AnalyticalConstantBasisFunction,
)
from model.ConstitutiveLaws.class_LinearElastic import LinearElastic
from model.PDEs.BalanceOfLinearMomentum import BalanceLinearMomentum
from model.Distributions.distribution_factory_selection import (
    distribution_factory_selection,
)
from model.ParentClasses.class_Priors_or_Posteriors_Handler import (
    DistributionHandler,
)


def clean_bf_args(field_options: dict) -> None:
    """Ensure 'bf_args' only contains allowed keys for a single field."""
    allowed_keys = [
        "flag_given_values",
        "given_values",
        "mask_given_values",
        "flag_norm",
    ]
    bf_args = field_options.setdefault("bf_args", {})
    for key in list(bf_args.keys()):
        if key not in allowed_keys:
            del bf_args[key]


def prepare_basis_function_options(
    field_options: dict, scale_x: float, scale_y: float
) -> dict:
    """
    Prepares and returns options for initializing a basis function class.
    This includes copying the options, setting bf_args, creating the bf_grid, and cleaning bf_args.
    """
    options = field_options.copy()
    options["bf_args"] = field_options
    bfs_grid = regular_2D_mesh(
        field_options["n_bfs_s1"],
        field_options["n_bfs_s2"],
        scale_x=scale_x,
        scale_y=scale_y,
        on_boundary=True,
    )
    options["bf_grid"] = bfs_grid
    clean_bf_args(options)
    return options


def load_displacement_observations(
    load_u_path: str,
    nodes_s1: int,
    nodes_s2: int,
    device: torch.device,
    default_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Loads the (non-noisy) displacement observation data and interpolates it to the desired grid size.
    Returns:
        u_true: torch.Tensor of shape [2, nodes_s1, nodes_s2] on the correct device and dtype.
    """
    u_true_original = torch.load(load_u_path, map_location=device)
    if u_true_original.ndim == 1:
        if u_true_original.numel() == nodes_s1 * nodes_s2 * 2:
            u_true_original = (
                u_true_original.to(device=device, dtype=default_dtype).reshape(2, -1).T
            )
            return u_true_original  # already in correct shape
        if (
            (u_true_original.numel() / 2) ** 0.5
        ).is_integer():  # it is a square mesh, but not the correct one.
            side_length = int((u_true_original.numel() / 2) ** 0.5)
            u_true_original = u_true_original.to(
                device=device, dtype=default_dtype
            ).reshape(2, side_length, side_length)
    # Ensure shape is [2, H, W] for interpolation
    if u_true_original.ndim == 2:
        u_true_original = u_true_original.unsqueeze(0)
    # Interpolate to desired size
    u_true_interp = torch.nn.functional.interpolate(
        u_true_original.unsqueeze(0).unsqueeze(0),  # add batch dim
        size=(2, nodes_s1, nodes_s2),
        mode="trilinear" if u_true_original.shape[0] == 2 else "bilinear",
        align_corners=True,
    ).squeeze()
    u_true = u_true_interp.to(device=device, dtype=default_dtype)
    return u_true.flatten(
        start_dim=-2, end_dim=-1
    ).T  # standard is a flattend shape [num_nodes, 2]


def load_stress_reference(
    load_stress_path: str | None,
    n_bfs_s1: int,
    n_bfs_s2: int,
    device: torch.device,
    default_dtype: torch.dtype,
    nodesMap: torch.Tensor,
) -> torch.Tensor:
    """
    Loads the reference stress data and interpolates it to the basis function grid size.
    Returns:
        sigma_true: stress field interpolated and mapped to triangles, shape [num_triangles, 4]
    """

    if load_stress_path is not None:
        stress = torch.load(load_stress_path, map_location=device)
        stress = torch.tensor(stress, device=device, dtype=default_dtype)
        stress = stress.squeeze()
        stress = stress.flatten(start_dim=-2, end_dim=-1)
        return stress
    else:
        return torch.zeros(
            (nodesMap.shape[0], 4), device=device, dtype=default_dtype
        )  # return zero stress if no path provided

    #     if stress.squeeze()
    #     # Interpolate to bf grid size
    #     stress_interp = torch.nn.functional.interpolate(
    #         stress.unsqueeze(0).unsqueeze(0),
    #         size=(4, n_bfs_s1, n_bfs_s2),
    #         mode="trilinear",
    #         align_corners=True,
    #     ).squeeze()
    #     stress_2D = torch.tensor(stress_interp, device=device, dtype=default_dtype)
    # else:
    #     stress_2D = torch.zeros(
    #         4, n_bfs_s1, n_bfs_s2, device=device, dtype=default_dtype
    #     )
    # stress_at_nodes = stress_2D.flatten(start_dim=-2, end_dim=-1).T
    # return node_to_triangle_values(stress_at_nodes, nodesMap)


def setup_observation_locations(
    YToField, n_obs_s1: int, n_obs_s2: int, obs_positions_flat: torch.Tensor
) -> None:
    """
    Sets up observation node flags or precomputes locations for YToField.
    """
    my_dim_y = YToField.bf_grid.shape[
        -1
    ]  # or use n_bfs_s1/n_bfs_s2 if more appropriate
    if my_dim_y * my_dim_y == n_obs_s1 * n_obs_s2:
        YToField.flag_obs_nodes = True
    else:
        YToField._location_precompute(obs_positions_flat)


def initialize_constitutive_law(
    MaterialModel: str,
    XToField: AnalyticalConstantBasisFunction,
    YToField: AnalyticalLinearBasisFunction,
    nu: float,
    flag_E_plus_one: bool,
) -> LinearElastic:
    """
    Initializes the constitutive law based on the material model.
    Returns:
        constitutive_law: the constitutive law object
    """
    constitutive_options = {
        "nu": nu,
        "XToField": XToField,
        "YToField": YToField,
        "flag_E_plus_one": flag_E_plus_one,
    }

    if MaterialModel == "LinearElastic":
        constitutive_law = LinearElastic(constitutive_options)
    else:
        raise ValueError(f"MaterialModel '{MaterialModel}' not known.")

    return constitutive_law


def initialize_pde(
    ThetaToField: AnalyticalLinearBasisFunction,
    XiToField,
    args_rhs: dict,
    device: torch.device,
) -> BalanceLinearMomentum:
    """
    Initializes the PDE object.
    Returns:
        pde: the PDE object
    """
    # Convert args_rhs to torch tensors on the correct device
    args_rhs_torch = {k: torch.tensor(v, device=device) for k, v in args_rhs.items()}

    pde_options = {
        "args_rhs": args_rhs_torch,
    }

    pde = BalanceLinearMomentum(
        ThetaToField=ThetaToField,
        XiToField=XiToField,
        args_rhs=args_rhs_torch,
        options=pde_options,
    )

    return pde


def initialize_pde_harmonic(
    ThetaToField: AnalyticalLinearBasisFunction,
    XiToField: AnalyticalConstantBasisFunctionTensor,
    YToField: AnalyticalLinearBasisFunction,
    rho: float,
    frequency: float,
    args_rhs: dict,
    device: torch.device,
) -> HarmonicBalanceOfLinearMomentum:
    """
    Initializes the harmonic PDE object.
    Returns:
        pde: the PDE object
    """
    # Convert args_rhs to torch tensors on the correct device
    args_rhs_torch = {k: torch.tensor(v, device=device) for k, v in args_rhs.items()}

    pde_options = {
        "args_rhs": args_rhs_torch,
    }

    pde = HarmonicBalanceOfLinearMomentum(
        ThetaToField=ThetaToField,
        XiToField=XiToField,
        YToField=YToField,
        args_rhs=args_rhs_torch,
        options=pde_options,
        rho=rho,
        omega=frequency * 2 * 3.14159,
    )

    return pde


def create_stress_calculation_function(constitutive_law: LinearElastic) -> Callable:
    """
    Creates a stress calculation function that wraps the constitutive law.
    This function will be used by the likelihood for constitutive model evaluation.
    """

    def stress_calc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return constitutive_law.calculate_stress(x=x, y=y)

    return stress_calc


def create_prior(
    handler: DistributionHandler,
    prior_kind: str,
    prior_options: dict,
    prior_dimension: int,
    additional_options: dict | None = None,
) -> DistributionHandler:
    """
    Initializes a prior using distribution_factory_selection and adds it to the handler.
    Returns the updated handler.
    """

    # add additional dimension information to prior options
    for key in ["mean", "covariance"]:
        if key in prior_options and isinstance(prior_options[key], dict):
            prior_options[key]["dim"] = prior_dimension
    prior_options["dim"] = prior_dimension
    # if additional options are provided, merge them into prior_options
    prior_options = dict(
        (
            merge(prior_options, additional_options)
            if additional_options
            else prior_options
        )
    )

    # create the prior distribution using the factory
    prior = distribution_factory_selection(prior_kind, prior_options)
    # add the prior to the handler
    handler.add(
        prior_options["name"],
        prior,
        SpecialInput=prior_options.get("SpecialInput", None),
        closedFormUpdateAvailable=prior_options.get("closedFormUpdateAvailable", False),
        expectation_active=prior_options.get("expectation_active", False),
    )
    return handler


def prepare_theta_options(
    theta_options: list,
    BasisFunctionGrid: torch.Tensor,
    NumUnknownParameters: int | torch.Tensor,
    MaskWithGivenValuesForBasisFunctionGrid: list[bool] | None = None,
) -> list[dict]:
    """
    Updates theta_options with necessary grid info (but does not create the DataLoader).
    Returns the updated theta_options.
    """
    max_x = float(BasisFunctionGrid[0].max())
    min_x = float(BasisFunctionGrid[0].min())
    max_y = float(BasisFunctionGrid[1].max())
    min_y = float(BasisFunctionGrid[1].min())

    for opt in theta_options:
        opt["dim_theta"] = NumUnknownParameters
        opt.setdefault("kwargs", {})
        opt["kwargs"]["n_bfs"] = BasisFunctionGrid.numel()
        opt["kwargs"]["bf_grid"] = BasisFunctionGrid
        opt["kwargs"]["mask_given_values"] = MaskWithGivenValuesForBasisFunctionGrid
        if opt["kind"] in ["rbf"]:
            opt["kwargs"]["max_x"] = max_x
            opt["kwargs"]["min_x"] = min_x
            opt["kwargs"]["max_y"] = max_y
            opt["kwargs"]["min_y"] = min_y

    return theta_options


def create_likelihood(
    likelihoods: DistributionHandler,
    likelihood_kind: str,
    likelihood_options: dict,
    additional_options: dict | None = None,
) -> DistributionHandler:
    """
    Initializes a likelihood using distribution_factory_selection and adds it to the handler.
    Returns the updated handler.
    """
    # Merge additional options if provided
    if additional_options:
        likelihood_options = dict(
            merge(additional_options, likelihood_options)
        )  # inverse order, as it will internally pickle the second dict, which is not possible for a torch.utils.DataLoader with a Generator specified. Dont ask me.

    # Create the likelihood distribution using the factory
    likelihood = distribution_factory_selection(likelihood_kind, likelihood_options)

    # Add the likelihood to the handler
    likelihoods.add(
        likelihood_options["name"],
        likelihood,
        SpecialInput=likelihood_options.get("SpecialInput", None),
        closedFormUpdateAvailable=likelihood_options.get(
            "closedFormUpdateAvailable", False
        ),
        expectation_active=likelihood_options.get("expectation_active", False),
    )

    return likelihoods


def add_input_dims_to_options_for_neural_net(
    options: dict, dict_of_dims: dict[str, int]
) -> dict:
    """
    If options["mean"]["type"] == "neural_net", adds options["mean"]["input_dims"] based on input_names.
    Returns the modified options dict.
    Throws ValueError if required keys are missing or unexpected input_names are found.
    """
    if (
        "mean" not in options
        or not isinstance(options["mean"], dict)
        or (
            options["mean"].get("type") != "neural_net"
            and options["mean"].get("type") != "res_net"
        )
    ):
        return options

    input_names = options["mean"].get("input_names")
    if not isinstance(input_names, list):
        raise ValueError("options['mean']['input_names'] must be a list of strings.")

    valid_names = {"x", "y", "xi"}
    input_dims = []
    for name in input_names:
        if name not in valid_names:
            raise ValueError(
                f"Unexpected input name '{name}' in options['mean']['input_names']."
            )
        if name not in dict_of_dims:
            raise ValueError(f"Missing key '{name}' in dict_of_dims dict.")
        dim = dict_of_dims[name]
        input_dims.append(dim)

    options["mean"]["input_dims"] = input_dims
    return options


def add_input_dims_to_options_for_neural_net_with_function(
    options: dict,
    transformation_fn: Callable[..., torch.Tensor],
    input_dim: int,
):
    """Adds the transformation function and the input dimensions (equal to the output dimensions of the transformation function) to the options dict for a neural net mean function.

    Args:
        options (dict): _description_
        transformation_fn (Callable[..., torch.Tensor]): _description_
        input_dim (int): _description_
    """
    if (
        "mean" not in options
        or not isinstance(options["mean"], dict)
        or (
            options["mean"].get("type") != "neural_net_with_function"
            and options["mean"].get("type") != "res_net_with_function"
            and options["mean"].get("type") != "res_add_with_function"
        )
    ):
        return options
    options["mean"]["transformation_fn"] = transformation_fn
    options["mean"]["input_dims"] = [input_dim]
    return options


def create_posterior(
    handler: DistributionHandler,
    posterior_kind: str,
    posterior_options: dict,
    posterior_dimension: int,
    additional_options: dict | None = None,
) -> DistributionHandler:
    """
    Initializes a posterior using distribution_factory_selection and adds it to the handler.
    Returns the updated handler.
    """

    # add additional dimension information to posterior options
    for key in ["mean", "covariance", "gamma"]:
        if key in posterior_options and isinstance(posterior_options[key], dict):
            posterior_options[key]["dim"] = posterior_dimension
    posterior_options["dim"] = posterior_dimension
    # if additional options are provided, merge them into posterior_options
    posterior_options = (
        dict(merge(posterior_options, additional_options))
        if additional_options
        else posterior_options
    )

    # create the posterior distribution using the factory
    posterior = distribution_factory_selection(posterior_kind, posterior_options)
    # add the posterior to the handler
    handler.add(
        posterior_options["name"],
        posterior,
        SpecialInput=posterior_options.get("SpecialInput", None),
        closedFormUpdateAvailable=posterior_options.get(
            "closedFormUpdateAvailable", False
        ),
        expectation_active=posterior_options.get("expectation_active", False),
    )
    return handler


def load_checkpoint_file(checkpoint_path: str, device: torch.device):
    """
    Loads a checkpoint from the specified path.
    Returns:
        - checkpoint: the loaded checkpoint dictionary
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")

    print("Loading parameters from {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def apply_checkpoint(
    checkpoint: dict,
    posterior: DistributionHandler,
    dont_load: list[str],
    u_hat: torch.Tensor,
) -> Tuple[DistributionHandler, torch.Tensor]:
    posterior.load_state_dict(checkpoint, dont_load=dont_load)

    if "u_hat" in checkpoint:
        u_hat = checkpoint["u_hat"].clone().detach()
    else:
        warnings.warn(
            "u_hat not found in loaded parameters. Using provided u_hat instead."
        )

    return posterior, u_hat
