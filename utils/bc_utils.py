import torch


def set_dirichlet_bc(
    field_options, apply_bc: bool, conditions: list, dims: int = 2, field_name="field"
):
    """
    Args:
        field_options: dict with n_bfs_s1, n_bfs_s2 (for 2D), etc.
        apply_bc: bool, whether to apply BCs
        conditions: list of dicts, each with keys 'boundary', 'dim' (int or list), 'value'
        dims: number of dimensions (default 2)
        field_name: str, for debugging

    Boundary types supported:
        - "left", "right", "bottom", "top": full edge boundaries
        - "TLCorner": Top-Left corner (single node)
        - "TRCorner": Top-Right corner (single node)
        - "BLCorner": Bottom-Left corner (single node)
        - "BRCorner": Bottom-Right corner (single node)

    Returns:
        field_options: updated dict with mask_given_values, given_values, flag_given_values
    """
    if not apply_bc or not conditions:
        field_options["flag_given_values"] = False
        return field_options

    n_bfs_s1 = field_options["n_bfs_s1"]
    n_bfs_s2 = field_options["n_bfs_s2"]
    mask = torch.zeros(dims, n_bfs_s1, n_bfs_s2, dtype=torch.bool)
    values = [[] for _ in range(dims)]

    for cond in conditions:
        boundary = cond["boundary"]
        dims_to_set = cond["dim"]
        value = cond["value"]
        # Support both int and list for dims
        if isinstance(dims_to_set, int):
            dims_to_set = [dims_to_set]
        for d in dims_to_set:
            if boundary == "left":
                mask[d, :, 0] = True
                values[d].extend([value] * n_bfs_s2)
            elif boundary == "right":
                mask[d, :, -1] = True
                values[d].extend([value] * n_bfs_s2)
            elif boundary == "bottom":
                mask[d, 0, :] = True
                values[d].extend([value] * n_bfs_s1)
            elif boundary == "top":
                mask[d, -1, :] = True
                values[d].extend([value] * n_bfs_s1)
            # Corner boundary conditions
            elif boundary == "TLCorner":  # Top-Left corner
                mask[d, -1, 0] = True
                values[d].append(value)
            elif boundary == "TRCorner":  # Top-Right corner
                mask[d, -1, -1] = True
                values[d].append(value)
            elif boundary == "BLCorner":  # Bottom-Left corner
                mask[d, 0, 0] = True
                values[d].append(value)
            elif boundary == "BRCorner":  # Bottom-Right corner
                mask[d, 0, -1] = True
                values[d].append(value)
            else:
                raise ValueError(f"Unknown boundary '{boundary}' for {field_name}")

    # Flatten mask and values for storage
    field_options["mask_given_values"] = mask.flatten()
    # Concatenate all values for all dims, then select only those where mask is True
    all_values = []
    for d in range(dims):
        all_values.extend(values[d])
    all_values = torch.tensor(all_values, dtype=torch.get_default_dtype())
    num_given = mask.sum()
    # Only keep as many values as there are True in the mask
    field_options["given_values"] = all_values[:num_given]
    field_options["flag_given_values"] = True
    return field_options


def create_boundary_conditions_general(
    YToField_options, ThetaToField_options, bc_config: dict, dims: int = 2
):
    """
    bc_config example:
    {
        "Y": {
            "apply": True,
            "conditions": [
                {"boundary": "bottom", "dim": 0, "value": 1.0},  # dim can be int or list
                {"boundary": "left", "dim": [1], "value": 2.0},
                {"boundary": "TLCorner", "dim": 1, "value": 0.0},  # Top-left corner
            ]
        },
        "Theta": {
            "apply": True,
            "conditions": [
                {"boundary": "bottom", "dim": [0, 1], "value": 0.0},
                {"boundary": "BRCorner", "dim": 0, "value": 0.0},  # Bottom-right corner
            ]
        }
    }
    """
    # Y boundary conditions
    if "Y" in bc_config:
        YToField_calc_options = set_dirichlet_bc(
            YToField_options,
            bc_config["Y"]["apply"],
            bc_config["Y"]["conditions"],
            dims=dims,
            field_name="Y",
        )
    else:
        YToField_calc_options = YToField_options.copy()
        YToField_calc_options["flag_given_values"] = False

    # Theta boundary conditions
    if "Theta" in bc_config:
        ThetaToField_calc_options = set_dirichlet_bc(
            ThetaToField_options,
            bc_config["Theta"]["apply"],
            bc_config["Theta"]["conditions"],
            dims=dims,
            field_name="Theta",
        )
    else:
        ThetaToField_calc_options = ThetaToField_options.copy()
        ThetaToField_calc_options["flag_given_values"] = False

    return YToField_calc_options, ThetaToField_calc_options
