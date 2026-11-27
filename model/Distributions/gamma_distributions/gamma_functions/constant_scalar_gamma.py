import torch
import math
from utils.torch_funcs.function_reparameterizedGamma import reparameterized_gamma

class ConstantScalarGamma(torch.nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = float(a)
        self.b = float(b)

    def sample(self, num_samples=None, **kwargs):
        sample_shape = torch.Size([num_samples]) if num_samples is not None else torch.Size()
        a = torch.tensor(self.a)
        b = torch.tensor(self.b)
        return reparameterized_gamma(a, b, sample_shape)

    def log_prob(self, x, **kwargs):
        a = self.a
        b = self.b
        logp = a * math.log(b) - math.lgamma(a) + (a - 1) * torch.log(x) - b * x
        return logp.sum(dim=-1) if x.ndim > 0 else logp

    def entropy(self, **kwargs):
        a = self.a
        b = self.b
        entropy = a - math.log(b) + math.lgamma(a) + (1 - a) * torch.digamma(torch.tensor(a))
        return entropy

    def mean(self, **kwargs):
        return self.a / self.b