import torch

def create_neighbours_mask(nodes_map: list | torch.Tensor) -> torch.Tensor:
    """
    Creates a neighbour difference mask matrix for a regular triangular grid.

    Given a list or tensor of node indices (nodes_map), this function constructs a matrix
    of shape (num_neighbour_pairs, num_elements), where each row corresponds to a pair of
    neighbouring elements. In each row, the entry for one element is +1, the entry for its
    neighbour is -1, and all other entries are 0. This mask can be used to efficiently compute
    differences between neighbouring elements in the grid.

    Args:
        nodes_map (list or torch.Tensor): List or tensor of shape (num_elements, nodes_per_element)
            containing the node indices for each element in the grid.

    Returns:
        torch.Tensor: A mask matrix of shape (num_neighbour_pairs, num_elements), where each row
            encodes a pair of neighbouring elements with +1 and -1.
    """
    elements = nodes_map
    num_elements = len(elements)
    mask_matrix = None

    for elem_idx in range(num_elements):
        # Repeat the current element's nodes for comparison with subsequent elements
        current_elem_nodes = elements[elem_idx].repeat(num_elements - (elem_idx + 1), 1)
        # Get the nodes of all subsequent elements
        subsequent_elem_nodes = elements[(elem_idx + 1):]
        # Concatenate node indices for pairwise comparison
        combined_nodes = torch.cat([current_elem_nodes, subsequent_elem_nodes], dim=1)
        # Sort node indices to facilitate comparison
        sorted_nodes = torch.sort(combined_nodes, dim=1)[0]
        # Count the number of unique nodes per row (shared nodes indicate neighbours)
        num_unique_nodes = sorted_nodes.size(1) - ((sorted_nodes[:, 1:] != sorted_nodes[:, :-1]).sum(dim=1) + 1)
        # Identify indices of neighbouring elements (sharing more than one node)
        neighbour_indices = (elem_idx + 1) + torch.where(num_unique_nodes.gt(1))[0]
        num_neighbours = len(neighbour_indices)
        # Create a temporary mask: +1 for current element, -1 for each neighbour
        temp_mask = torch.zeros(num_elements, num_neighbours)
        temp_mask[elem_idx, :] = 1
        temp_mask[neighbour_indices] = torch.eye(num_neighbours) * (-1)
        # Concatenate to the overall mask matrix
        if mask_matrix is None:
            mask_matrix = temp_mask
        else:
            mask_matrix = torch.cat((mask_matrix, temp_mask), dim=1)

    return mask_matrix.permute(1, 0)