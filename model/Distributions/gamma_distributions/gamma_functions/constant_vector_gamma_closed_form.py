import torch
from utils.torch_funcs.function_reparameterizedGamma import reparameterized_gamma
from utils.torch_funcs.function_make_torch_tensor import make_torch_tensor

class ConstantVectorGammaClosedForm(torch.nn.Module):
    def __init__(self, a, b, a_0, b_0, variant="gaussian", intermediate_key="r_c", dim=1):
        super().__init__()
        self.register_buffer('a', make_torch_tensor(a, torch.Size([dim])))
        self.register_buffer('b', make_torch_tensor(b, torch.Size([dim])))
        self.a_0 =  make_torch_tensor(a_0, torch.Size([dim]))
        self.b_0 =  make_torch_tensor(b_0, torch.Size([dim]))
        self.variant = variant
        self.intermediate_key = intermediate_key

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

    def closedFormUpdate(self, samples, intermediates):
        r_c = intermediates[self.intermediate_key].clone().detach()
        if self.variant == "gaussian":
            if r_c.ndim == 1:
                r_c = r_c.unsqueeze(-1)
            r_c_mean_sqr = r_c.pow(2).mean(dim=-2)
            self.a = self.a_0 + 0.5
            self.b = self.b_0 + r_c_mean_sqr / 2
        elif self.variant == "exponential":
            r_c_mean = r_c.mean(dim=-2)
            self.a = self.a_0 + 1.0
            self.b = self.b_0 + r_c_mean
        else:
            raise ValueError(f"Unknown variant: {self.variant}")