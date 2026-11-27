import torch
from model.ParentClasses.class_ParameterToField import ParameterToField
from utils.torch_funcs.function_check_inside_triangle import point_in_triangle


class AnalyticalLinearBasisFunction(ParameterToField):
    def __init__(self, options: dict) -> None:
        # Intermediate store for the shape functions

        self.flag_obs_nodes = False
        self.obs_element = None
        self.obs_xi = None

        super().__init__("grid", 2, options["bf_args"], options["bf_grid"])
        # get the triangular nodes
        self.triangShapeFuncsNodes, self.triangShapeFuncsNodes_deriv, self.Two_A = (
            self._create_set_of_bf()
        )

        # grid bounds
        self.max_x = self.bf_grid[-2, :, :].max()
        self.min_x = self.bf_grid[-2, :, :].min()
        self.max_y = self.bf_grid[-1, :, :].max()
        self.min_y = self.bf_grid[-1, :, :].min()

        # distance between grid points
        self.len_x = (self.max_x - self.min_x) / (self.bf_grid.size(dim=-2) - 1)
        self.len_y = (self.max_y - self.min_y) / (self.bf_grid.size(dim=-1) - 1)

        # get list of boundary elements
        (
            self.top_node_list,
            self.bottom_node_list,
            self.left_node_list,
            self.right_node_list,
        ) = self._get_boundary_elements()

        self.elements_per_node_list = self._get_elements_per_node_list()

    def eval_grad_s(
        self, parameter: torch.Tensor, apply_known_values: bool = True
    ) -> torch.Tensor:
        # if I have given values, I have to place them in the parameter vector
        if apply_known_values:
            parameter = self.apply_know_values(parameter)

        # from 1D [vx1, vx2, ..., vy1, vy2, ...] to 2D: N x [[vx1, vy1], [vx2, vy2], ...], where N is the number of elements
        parameter_local = self._rewrite_into_local(parameter)
        deriv = torch.einsum(
            "...ijk,kli->...ijl", parameter_local, self.triangShapeFuncsNodes_deriv
        )  # [..., Nele, 2, 3] x [3, 2, Nele] -> [..., Nele, 2, 2]

        return deriv  # element-wise

    def eval_grad_s_single(self, parameter: torch.Tensor) -> torch.Tensor:
        # HERE I ASSUME I GET ALL VALUES AND DONT NEED TO APPLY THE KNOWN VALUES
        # if len(parameter) != self.n_bfs/2:
        #     raise ValueError("The parameter vector is not the same length as the number of basis functions.")

        # from 1D [vx1, vx2, ..., vxn] to 2D: N x [[vx1, vy1], [vx2, vy2], ...], where N is the number of elements
        parameter_local = self._rewrite_into_local_single(parameter)
        deriv = torch.einsum(
            "...nd,din->...ni", parameter_local, self.triangShapeFuncsNodes_deriv
        )  # [..., Nele, 3] x [3, 2, Nele] -> [..., Nele, 2]
        deriv = deriv.squeeze()

        return deriv

    def eval_quadrature_points(
        self, parameter: torch.Tensor, apply_known_values: bool = True
    ) -> torch.Tensor:
        # if I have given values, I have to place them in the parameter vector
        if apply_known_values:
            parameter = self.apply_know_values(parameter)

        # from 1D [vx1, vx2, ..., vy1, vy2, ...] to 2D: N x [[vx1, vy1], [vx2, vy2], ...], where N is the number of elements
        parameter_local = self._rewrite_into_local(parameter)

        # integration is done via quadrature points
        # TODO: Note that this introduces a condition that the quadrature points are the same for all elements
        Q = torch.tensor(
            [[1 / 6, 1 / 6, 2 / 3], [1 / 6, 2 / 3, 1 / 6], [2 / 3, 1 / 6, 1 / 6]]
        )

        # calculate the values at the quadrature points
        u_Q = torch.einsum(
            "...jk, ki -> ...ij", parameter_local, Q
        )  # [..., Nele, 3, 2] x [3, 3] -> [..., Nele, 3, 2]

        return u_Q

    def eval(
        self, parameter: torch.Tensor, apply_known_values: bool = True
    ) -> torch.Tensor:
        if apply_known_values:
            parameter = self.apply_know_values(parameter)
        return parameter

    def eval_at_locations(
        self, parameter: torch.Tensor, flag_2D_output: bool = False
    ) -> torch.Tensor:
        x_global = self.apply_know_values(parameter)  # [..., N_bfs]
        if self.flag_obs_nodes:
            val_obs = x_global.view(
                -1, 2, self.bf_grid.size(dim=-2), self.bf_grid.size(dim=-1)
            ).flatten(start_dim=-2, end_dim=-1)  # [..., 2, N_bfs]
        else:
            # please run _location_precompute before this
            x_local = self._rewrite_into_local(x_global)
            x_ele = x_local[..., self.obs_element, :, :]
            val_obs = torch.einsum(
                "...ijk, ki-> ...ij", x_ele, self.obs_xi
            )  # [..., Nobs, 2, 3] x [3, Nobs] -> [..., Nobs, 2]
            val_obs = val_obs.permute(0, 2, 1)  # [..., 2, Nobs]
        if not flag_2D_output:
            val_obs = val_obs.flatten(start_dim=-2, end_dim=-1)
        return val_obs

    def _location_precompute(self, locations: torch.Tensor) -> None:
        # FIXME: This has some problem when point is in multiple elements (e.g. on edges or nodes)
        # init solution (-1 is for the case that the location is not in any triangle)
        result = torch.ones(locations.size(0), 3) * (-1)
        # find out in which triangle the location is
        for i, node in enumerate(self.nodes):
            mask = point_in_triangle(locations, node[:2, 0], node[:2, 1], node[:2, 2])
            if mask.sum() == 0:
                continue
            locs = locations[mask, :]
            node_num = i * torch.ones(locs.size(0))
            new_row = torch.concat([node_num.unsqueeze(1), locs], dim=1)
            result[mask] = new_row
        # check if all locations are inside a triangle
        if (result == -1).sum() > 0:
            raise ValueError("Some locations are not inside any triangle.")

        # get the element and the local coordinates
        xy1 = torch.concat((result[:, 1:], torch.ones(result.size(0), 1)), dim=1)
        element = result[:, 0].to(torch.int64)
        xi = torch.einsum(
            "ijk,kj -> ik", self.triangShapeFuncsNodes[:, :, element], xy1
        )  # [3 x Nobs] # 1/2A is already in shape functions

        self.obs_element = element
        self.obs_xi = xi

    def _create_set_of_bf(self) -> tuple:
        # create the coordinates of the triangles
        nodes_list = self.nodes
        nodesMap = self.nodesMap
        # Only stack if nodes_list is a list/tuple
        if isinstance(nodes_list, (list, tuple)):
            nodes = torch.stack(nodes_list, dim=0)
        else:
            nodes = nodes_list  # Already a tensor

        # order them counter-clockwise (or make sure that they are)
        # if det([[x1, x2, x3], [y1, y2, y3], [1, 1, 1]])
        # < 0 then clockwise (reorder)
        # > 0 then counter-clockwise (okay)
        # = 0 then colinear (error)
        # (https://math.stackexchange.com/questions/1324179/how-to-tell-if-3-connected-points-are-connected-clockwise-or-counter-clockwise)
        coords = nodes[:, :2, :]
        ones = torch.ones(coords.size(0), 1, coords.size(2))
        coords = torch.cat((coords, ones), dim=1)
        det = torch.det(coords)
        for i in range(det.size(0)):
            if det[i] < 0:
                nodes[i] = nodes[i, :, [0, 2, 1]]
                nodesMap[i] = nodesMap[i, [0, 2, 1]]

        # save for later use
        self.nodes = nodes
        self.nodesMap = nodesMap

        # short definitions
        x1 = nodes[:, 0, 0]
        x2 = nodes[:, 0, 1]
        x3 = nodes[:, 0, 2]
        y1 = nodes[:, 1, 0]
        y2 = nodes[:, 1, 1]
        y3 = nodes[:, 1, 2]

        # calculate triangle area(s)
        # 2A = (x_32 y_12 - x_12 y_32)
        # (Belytschko, Appendix 3, p.761)
        self.Two_A = (x3 - x2) * (y1 - y2) - (x1 - x2) * (y3 - y2)

        # tensor of the shape functions
        s1 = torch.stack([y2 - y3, x3 - x2, x2 * y3 - x3 * y2], dim=0)
        s2 = torch.stack([y3 - y1, x1 - x3, x3 * y1 - x1 * y3], dim=0)
        s3 = torch.stack([y1 - y2, x2 - x1, x1 * y2 - x2 * y1], dim=0)
        triangShapeFuncsNodes = (
            1 / self.Two_A * torch.stack([s1, s2, s3], dim=0)
        )  # Nele x 3 x 3
        # self.triangShapeFuncs = 1/Two_A * torch.tensor([[y2 - y3, x3 - x2, x2 * y3 - x3 * y2], [y3 - y1, x1 - x3, x3 * y1 - x1 * y3], [y1 - y2, x2 - x1, x1 * y2 - x2 * y1]])

        # tensor of the shape functions derivatives
        a1 = torch.stack([y2 - y3, x3 - x2], dim=0)
        a2 = torch.stack([y3 - y1, x1 - x3], dim=0)
        a3 = torch.stack([y1 - y2, x2 - x1], dim=0)
        triangShapeFuncsNodes_deriv = 1 / self.Two_A * torch.stack([a1, a2, a3], dim=0)
        # self.triangShapeFuncsNodes_deriv = 1/Two_A * torch.tensor([, [y3 - y1, x1 - x3], [y1 - y2, x2 - x1]])

        return triangShapeFuncsNodes, triangShapeFuncsNodes_deriv, self.Two_A

    def _rewrite_into_local(self, x: torch.Tensor) -> torch.Tensor:
        # x is a 1D tensor
        x2 = x.view(
            -1, 2, int(self.n_bfs / 2)
        )  # [v1x, v2x, ..., v1y, v2y, ...] -> [[v1x, v1y], [v2x, v2y], ...] -> # [..., 2, Nbfs]

        # get the local triangle coordinates
        x3 = x2[..., :, self.nodesMap]  # [..., Nele, 2, 3]
        x_new = x3.permute(0, 2, 1, 3)  # [..., Nele, 3, 2]
        return x_new

    def _rewrite_into_local_single(self, x: torch.Tensor) -> torch.Tensor:
        x_new = x[..., self.nodesMap]  # [..., Nele, 3]
        # x_new = x2.unsqueeze(-1) # [..., Nele, 3, 1]
        return x_new

    def _get_boundary_elements(self) -> tuple:
        nodes = self.nodes[:, :2, :]
        # get the boundary elements
        # top
        top_mask_nodes = (nodes[:, 1, :] - self.max_y) > -1e-4
        top_node_list = self._get_BC_node_list(top_mask_nodes)

        # bottom
        bottom_mask = (nodes[:, 1, :] - self.min_y) < 1e-4
        bottom_node_list = self._get_BC_node_list(bottom_mask)

        # left
        left_mask = (nodes[:, 0, :] - self.min_x) < 1e-4
        left_node_list = self._get_BC_node_list(left_mask)

        # right
        right_mask = (nodes[:, 0, :] - self.max_x) > -1e-4
        right_node_list = self._get_BC_node_list(right_mask)

        return top_node_list, bottom_node_list, left_node_list, right_node_list

    def _get_BC_node_list(self, mask_nodes: torch.Tensor) -> torch.Tensor:
        top_mask_elements = mask_nodes.sum(dim=1) == 2
        top_mask_nodes_of_elements = mask_nodes * top_mask_elements.repeat(3, 1).T
        top_element_nodes = top_mask_nodes_of_elements * self.nodesMap
        top_element_nodes[
            ~top_mask_nodes_of_elements
        ] = -1  # this is so I dont count zeros later
        top_sum_nodes = torch.zeros(
            self.n_bfs // 2
        )  # I have to divide by 2 because I have x and y, but only one set of nodes
        for i in range(len(top_sum_nodes)):
            top_sum_nodes[i] = torch.sum(top_element_nodes == i)
        return top_sum_nodes

    def _get_elements_per_node_list(self) -> torch.Tensor:
        elements_per_node_list = torch.zeros(self.n_bfs // 2)
        for i in range(self.n_bfs // 2):
            elements_per_node_list[i] = torch.sum(self.nodesMap == i)
        return elements_per_node_list
