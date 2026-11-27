from matplotlib.tri import Triangulation
import torch
from typing import Tuple


def count_triangulation_fields_in_vector(
    vector: torch.Tensor,
    triangulation: Triangulation,
) -> Tuple[int, int, str]:
    vector_size = vector.size()[-1]
    triangulation_number_of_elements = triangulation.triangles.shape[0]
    triangulation_number_of_nodes = triangulation.x.shape[0]

    number_element_fields = vector_size / triangulation_number_of_elements
    number_node_fields = vector_size / triangulation_number_of_nodes

    if number_element_fields.is_integer() and not number_node_fields.is_integer():
        return (
            int(number_element_fields),
            int(triangulation_number_of_elements),
            "element",
        )
    elif number_node_fields.is_integer() and not number_element_fields.is_integer():
        return int(number_node_fields), int(triangulation_number_of_nodes), "node"
    else:
        raise ValueError(
            f"Vector size {vector_size} is not compatible with the triangulation (number of elements: {triangulation_number_of_elements}, number of nodes: {triangulation_number_of_nodes})."
        )
