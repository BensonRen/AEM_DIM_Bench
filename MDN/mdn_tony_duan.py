import torch
import torch.nn as nn
from torch.distributions import Normal, OneHotCategorical

# Implementation adapted from:
#  https://github.com/tonyduan/mdn/blob/master/mdn/models.py

class MDN(nn.Module):
    """
    Mixture density network.
    [ Bishop, 1994 ]
    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, flags):
        super().__init__()
        self.pi_network = CategoricalNetwork(flags)
        self.normal_network = MixtureDiagNormalNetwork(flags)

    def forward(self, x):
        return self.pi_network(x), self.normal_network(x)

    def loss(self, x, y):
        pi, normal = self.forward(x)
        loglik = normal.log_prob(y.unsqueeze(1).expand_as(normal.loc))
        loglik = torch.sum(loglik, dim=2)
        loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=1)
        return loss

    def sample(self, x):
        pi, normal = self.forward(x)
        samples = torch.sum(pi.sample().unsqueeze(2) * normal.sample(), dim=1)
        return samples


class MixtureDiagNormalNetwork(nn.Module):

    def __init__(self, flags):
        super().__init__()
        in_features, out_features, num_gaussians = flags.linear[0], flags.linear[-1], flags.num_gaussian
        self.n_components = num_gaussians
        self.mu = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear[:-2]):
            self.mu.append(nn.Linear(fc_num, flags.linear[ind + 1]))
            # self.mu.append(nn.BatchNorm1d(flags.linear[ind + 1]))
            self.mu.append(nn.ELU())
        self.mu.append(nn.Linear(flags.linear[-2], 2*out_features*num_gaussians))
        self.network = nn.Sequential(*self.mu)
        # Original implementation
        #  if hidden_dim is None:
        #     hidden_dim = in_dim
        # self.network = nn.Sequential(
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.ELU(),
        #     nn.Linear(hidden_dim, 2 * out_dim * n_components),
        # )

    def forward(self, x):
        params = self.network(x)
        mean, sd = torch.split(params, params.shape[1] // 2, dim=1)
        mean = torch.stack(mean.split(mean.shape[1] // self.n_components, 1))
        sd = torch.stack(sd.split(sd.shape[1] // self.n_components, 1))
        return Normal(mean.transpose(0, 1), torch.exp(sd).transpose(0, 1))

class CategoricalNetwork(nn.Module):

    def __init__(self, flags):
        super().__init__()
        in_features, out_features, num_gaussians = flags.linear[0], flags.linear[-1], flags.num_gaussian
        self.pi = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear[:-2]):
            self.pi.append(nn.Linear(fc_num, flags.linear[ind + 1]))
            # self.pi.append(nn.BatchNorm1d(flags.linear[ind + 1]))
            self.pi.append(nn.ELU())
        self.pi.append(nn.Linear(flags.linear[-2], num_gaussians))
        self.network = nn.Sequential(*self.pi)
        
        # Original implementation
        # self.network = nn.Sequential(
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.ELU(),
        #     nn.Linear(hidden_dim, out_dim)
        # )

    def forward(self, x):
        params = self.network(x)
        return OneHotCategorical(logits=params)