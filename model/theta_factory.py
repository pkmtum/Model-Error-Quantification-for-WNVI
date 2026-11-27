import torch


class ThetaBuilderBase:
    def __init__(self, dim_theta, num_samples, **kwargs):
        self.dim_theta = dim_theta
        self.num_samples = num_samples
        self.kwargs = kwargs

    def build(self):
        raise NotImplementedError


class RandomThetaBuilder(ThetaBuilderBase):
    def build(self):
        mean = self.kwargs.get("mean", 0.0)
        std = self.kwargs.get("std", 1.0)
        return torch.randn(self.num_samples, self.dim_theta) * std + mean


class FullThetaBuilder(ThetaBuilderBase):
    def build(self):
        # Stack num_samples eye matrices vertically (one after another)
        eye = torch.eye(self.dim_theta)
        return torch.vstack([eye for _ in range(self.num_samples)])


class RandomFullThetaBuilder(ThetaBuilderBase):
    def build(self):
        eye = torch.eye(self.dim_theta)
        idx = torch.randint(0, self.dim_theta, (self.num_samples,))
        return eye[idx]


def _sample_centers_and_radii(bf_grid, num_samples, min_rad, max_rad, radius_type):
    bfs_per_row = bf_grid.size()[1]
    bfs_per_col = bf_grid.size()[2]
    x = torch.randint(0, bfs_per_row, (num_samples,))
    y = torch.randint(0, bfs_per_col, (num_samples,))
    center = bf_grid[:, x, y]
    if radius_type == "linear":
        radius = torch.rand(num_samples) * (max_rad - min_rad) + min_rad
    elif radius_type == "log":
        radius = torch.exp(
            torch.rand(num_samples) * (torch.log(max_rad) - torch.log(min_rad))
            + torch.log(min_rad)
        )
    else:
        raise ValueError("Radius type must be 'linear' or 'log'")
    return center, radius


def _apply_flip_and_mask(theta, points, n_bfs, num_samples, BC_mask):
    flip1 = torch.randint(0, 2, (num_samples,)).to(torch.bool)
    flip2 = torch.randint(0, 2, (num_samples,)).to(torch.bool)
    flip3 = ~(flip1 | flip2)
    flip1 = flip1 | flip3
    flip2 = flip2 | flip3
    theta[n_bfs // 2 :, flip1] = points[:, flip1]
    theta[: n_bfs // 2, flip2] = points[:, flip2]
    if BC_mask is not None:
        theta = theta[~BC_mask, :]
    theta = theta.to(torch.get_default_dtype())
    theta = theta.permute(1, 0)
    return theta


class CircularThetaBuilder(ThetaBuilderBase):
    def build(self):
        bf_grid = self.kwargs["bf_grid"]
        n_bfs = self.kwargs["n_bfs"]  # == bf_grid.numel()
        min_rad = self.kwargs["min_rad"]
        max_rad = self.kwargs["max_rad"]
        radius_type = self.kwargs.get("radius_type", "linear")
        BC_mask = self.kwargs.get("mask_given_values", None)

        center, radius = _sample_centers_and_radii(
            bf_grid, self.num_samples, min_rad, max_rad, radius_type
        )
        dist_center = torch.sub(
            bf_grid.repeat(self.num_samples, 1, 1, 1).permute(2, 3, 1, 0), center
        )
        sq_dist_center = torch.pow(dist_center, 2)
        d = sq_dist_center.sum(dim=2)
        mask = d <= radius**2
        points = mask.flatten(start_dim=0, end_dim=1)
        theta = torch.zeros(n_bfs, self.num_samples, dtype=torch.bool)  #
        return _apply_flip_and_mask(theta, points, n_bfs, self.num_samples, BC_mask)


class RectangleThetaBuilder(ThetaBuilderBase):
    def build(self):
        bf_grid = self.kwargs["bf_grid"]
        n_bfs = self.kwargs["n_bfs"]
        min_rad = self.kwargs["min_rad"]
        max_rad = self.kwargs["max_rad"]
        radius_type = self.kwargs.get("radius_type", "linear")
        BC_mask = self.kwargs.get("mask_given_values", None)

        center, radius = _sample_centers_and_radii(
            bf_grid, self.num_samples, min_rad, max_rad, radius_type
        )
        dist_center = torch.sub(
            bf_grid.repeat(self.num_samples, 1, 1, 1).permute(2, 3, 1, 0), center
        )
        abs_dist_center = abs(dist_center)
        mask_x = abs_dist_center[:, :, 0, :] <= radius[None, None, :]
        mask_y = abs_dist_center[:, :, 1, :] <= radius[None, None, :]
        mask = mask_x & mask_y
        points = mask.flatten(start_dim=0, end_dim=1)
        theta = torch.zeros(n_bfs, self.num_samples, dtype=torch.bool)
        return _apply_flip_and_mask(theta, points, n_bfs, self.num_samples, BC_mask)


class RBFThetaBuilder(ThetaBuilderBase):
    def build(self):
        n_bfs = self.kwargs["n_bfs"]
        bf_grid = self.kwargs["bf_grid"]
        min_rad = self.kwargs["min_rad"]
        max_rad = self.kwargs["max_rad"]
        BC_mask = self.kwargs.get("mask_given_values", None)
        if isinstance(self.kwargs["max_x"], torch.Tensor):
            max_x = self.kwargs["max_x"].clone().detach()
            min_x = self.kwargs["min_x"].clone().detach()
            max_y = self.kwargs["max_y"].clone().detach()
            min_y = self.kwargs["min_y"].clone().detach()
        else:
            max_x = torch.tensor(self.kwargs["max_x"])
            min_x = torch.tensor(self.kwargs["min_x"])
            max_y = torch.tensor(self.kwargs["max_y"])
            min_y = torch.tensor(self.kwargs["min_y"])

        center = torch.rand(self.num_samples, 2) * torch.stack(
            [max_x - min_x, max_y - min_y]
        ) + torch.stack([min_x, min_y])
        radius = torch.rand(self.num_samples) * (max_rad - min_rad) + min_rad
        epsilon = 1 / (radius)
        dist_center = torch.sub(
            bf_grid.repeat(self.num_samples, 1, 1, 1).permute(2, 3, 1, 0), center.T
        )
        sq_dist_center = torch.pow(dist_center, 2)
        d = torch.sqrt(sq_dist_center.sum(dim=2))
        points_values = torch.exp(-((epsilon * d) ** 2))
        points = points_values.flatten(start_dim=0, end_dim=1)
        theta = torch.zeros(n_bfs, self.num_samples)
        return _apply_flip_and_mask(theta, points, n_bfs, self.num_samples, BC_mask)


class ThetaSetFactory:
    def __init__(self, kind, dim_theta, num_samples, **kwargs):
        self.kind = kind.lower()
        self.dim_theta = dim_theta
        self.num_samples = num_samples
        self.kwargs = kwargs

    def get_builder(self):
        if self.kind == "random":
            return RandomThetaBuilder(self.dim_theta, self.num_samples, **self.kwargs)
        elif self.kind == "full":
            return FullThetaBuilder(self.dim_theta, self.num_samples, **self.kwargs)
        elif self.kind == "random_full":
            return RandomFullThetaBuilder(
                self.dim_theta, self.num_samples, **self.kwargs
            )
        elif self.kind == "circular":
            return CircularThetaBuilder(self.dim_theta, self.num_samples, **self.kwargs)
        elif self.kind == "rectangle":
            return RectangleThetaBuilder(
                self.dim_theta, self.num_samples, **self.kwargs
            )
        elif self.kind == "rbf":
            return RBFThetaBuilder(self.dim_theta, self.num_samples, **self.kwargs)
        else:
            raise ValueError(f"Unknown theta builder kind: {self.kind}")

    def build(self):
        builder = self.get_builder()
        return builder.build()
