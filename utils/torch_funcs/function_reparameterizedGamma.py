import torch

def reparameterized_gamma(a, b, sample_shape=torch.Size()):
    """
    Reparameterized sampling from Gamma(a, b) using Marsaglia & Tsang (2000).
    Supports both a >= 1 and a < 1 via boosting trick.
    
    Parameters:
        a (Tensor): shape parameter (concentration), must be positive
        b (Tensor): rate parameter (inverse scale), must be positive
        sample_shape (torch.Size or tuple): number of samples to draw

    Returns:
        Tensor: samples with shape `sample_shape + broadcast(a.shape, b.shape)`
    """
    eps = 1e-8

    if torch.any(a <= 0):
        raise ValueError("Shape parameter 'a' must be strictly positive.")
    if torch.any(b <= 0):
        raise ValueError("Rate parameter 'b' must be strictly positive.")

    a = a.expand(torch.broadcast_shapes(a.shape, b.shape))
    b = b.expand_as(a)

    # Expand to sample shape
    if isinstance(sample_shape, tuple):
        sample_shape = torch.Size(sample_shape)
    full_shape = sample_shape + a.shape
    a_exp = a.expand(full_shape)
    b_exp = b.expand(full_shape)

    is_small = a_exp < 1.0
    is_large = ~is_small

    def sample_large(alpha, beta):
        d = alpha - 1.0 / 3.0
        c = 1.0 / torch.sqrt(9.0 * d)
        normal = torch.randn_like(alpha)
        v = (1.0 + c * normal) ** 3
        v = torch.clamp(v, min=eps)
        return d * v / beta

    def sample_small(alpha, beta):
        u = torch.rand_like(alpha)
        boosted = sample_large(alpha + 1.0, beta)
        return boosted * u.pow(1.0 / alpha)

    samples = torch.where(is_small, sample_small(a_exp, b_exp), sample_large(a_exp, b_exp))
    return samples

