import torch
from torch.utils.data import TensorDataset, DataLoader
from model.theta_factory import ThetaSetFactory

def make_theta_loader(builders, batch_size):
    """
    builders: list of dicts, each dict has keys:
        - kind: str
        - dim_theta: int
        - num_samples: int
        - kwargs: dict (optional)
    """
    all_thetas = []
    for builder in builders:
        kind = builder["kind"]
        dim_theta = builder["dim_theta"]
        num_samples = builder["num_samples"]
        kwargs = builder.get("kwargs", {})
        factory = ThetaSetFactory(kind, dim_theta, num_samples, **kwargs)
        thetas = factory.build()
        all_thetas.append(thetas)
    thetas = torch.cat(all_thetas, dim=0)
    dataset = TensorDataset(thetas)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=torch.get_default_device()), drop_last=True)