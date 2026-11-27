import torch
# import matplotlib.pyplot as plt
# from function_regular_mesh import regular_2D_mesh


def generate_triangle_field(triangle_coords, s_grid):
    # Extract triangle coordinates
    x1, y1 = triangle_coords[0]
    x2, y2 = triangle_coords[1]
    x3, y3 = triangle_coords[2]

    # Full grids
    X, Y = s_grid[0, :, :], s_grid[1, :, :]

    # Calculate barycentric coordinates
    denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    alpha = ((y2 - y3) * (X - x3) + (x3 - x2) * (Y - y3)) / denominator
    beta = ((y3 - y1) * (X - x3) + (x1 - x3) * (Y - y3)) / denominator
    gamma = 1 - alpha - beta
    alphadx = (y2 - y3) / denominator
    alphady = (x3 - x2) / denominator

    # Create fields
    field = torch.zeros_like(X)
    field_dx = torch.zeros_like(X)
    field_dy = torch.zeros_like(X)

    # Set values within the triangle
    eps = 1e-5
    condition = (alpha >= -eps) & (beta >= -eps) & (gamma >= -eps)

    # condition = ((alpha >= eps) & (beta >= eps) & (gamma >= eps)) | ((alpha >= -eps) & (beta >= -eps) & (gamma >= -eps))
    field[condition] = alpha[condition]
    field_dx[condition] = alphadx  # these are only scalars
    field_dy[condition] = alphady  # scalar

    return field, field_dx, field_dy


# # Generate example triangle coordinates
# triangle_coords = [(0, 0), (1, 0), (0.5, 1)]

# # Generate field
# s_grid = regular_2D_mesh(50, 50, on_boundary=True)
# field = generate_triangle_field(triangle_coords, s_grid)

# # Visualize field
# plt.imshow(field, origin='lower', extent=[0, 1, 0, 1])
# plt.colorbar()
# plt.show()


def get_triangular_nodes(mesh: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    Takes coordinates of BF center
    Returns:
        nodes: list of coordinates of nodes of triangular elements (and node number)
        nodesMap: mapping of nodes to triangular elements
    """
    nodes = []
    x = mesh[
        0
    ]  # note that in bf grid, "x" is changing with s2 and "y" with s1, contrary to what one miht think, below it is also the case. Fuck me.
    y = mesh[1]
    nodes_in_s2 = x.size(0)
    nodes_in_s1 = x.size(1)
    if nodes_in_s2 < 2 or nodes_in_s1 < 2:
        raise ValueError(
            f"Mesh is too small to form triangles. You gave {mesh.size()}, but expected at least (2, 2, 2)."
        )
    nodeID = torch.arange(0, x.size(0) * x.size(1), dtype=torch.int32).reshape(
        x.size(0), x.size(1)
    )
    for j in range(nodes_in_s1 - 1):
        for i in range(nodes_in_s2 - 1):
            # Define triangular elements by connecting adjacent nodes
            triangle1 = torch.tensor(
                [
                    [x[i, j], x[i + 1, j], x[i + 1, j + 1]],
                    [y[i, j], y[i + 1, j], y[i + 1, j + 1]],
                    [nodeID[i, j], nodeID[i + 1, j], nodeID[i + 1, j + 1]],
                ]
            )
            triangle2 = torch.tensor(
                [
                    [x[i, j], x[i + 1, j + 1], x[i, j + 1]],
                    [y[i, j], y[i + 1, j + 1], y[i, j + 1]],
                    [nodeID[i, j], nodeID[i + 1, j + 1], nodeID[i, j + 1]],
                ]
            )
            nodes.append(triangle1)
            nodes.append(triangle2)

    nodesMap = torch.stack(nodes)[:, 2, :].to(torch.int32)

    return nodes, nodesMap
