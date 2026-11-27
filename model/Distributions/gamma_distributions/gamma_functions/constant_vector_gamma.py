import torch
from utils.torch_funcs.function_reparameterizedGamma import reparameterized_gamma
from utils.torch_funcs.function_make_torch_tensor import make_torch_tensor

class ConstantVectorGamma(torch.nn.Module):
    def __init__(self, a, b, dim=1):
        super().__init__()
        self.a = make_torch_tensor(a, torch.Size([dim]))
        self.b =  make_torch_tensor(b, torch.Size([dim]))

    def sample(self, num_samples=None, **kwargs):
        sample_shape = torch.Size([num_samples]) if num_samples is not None else torch.Size()
        return reparameterized_gamma(self.a, self.b, sample_shape)

    def log_prob(self, x, **kwargs):
        a = self.a
        b = self.b
        logp = a * torch.log(b) - torch.lgamma(a) + (a - 1) * torch.log(x) - b * x
        return logp.sum(dim=-1)

    def entropy(self, **kwargs):
        a = self.a
        b = self.b
        entropy = a - torch.log(b) + torch.lgamma(a) + (1 - a) * torch.digamma(a)
        return entropy.sum()

    def mean(self, **kwargs):
        return self.a / self.b