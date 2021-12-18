#Sample Implementation for educational purposes
#For full implementation check out https://github.com/manujosephv/pytorch_tabular
# Adapted from 
# https://deep-and-shallow.com/2021/03/20/mixture-density-networks-probabilistic-regression-for-uncertainty-estimation/
import torch
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np
#from numpy.random.Generator import multivariate_normal
from torch.distributions.multivariate_normal import MultivariateNormal
ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)
LOG2PI = math.log(2 * math.pi)

class MDN(nn.Module):
    def __init__(self, flags):
        self.hparams = flags
        self.hparams.input_dim = flags.linear[0]
        self.hparams.sigma_bias_flag = True
        self.hparams.mu_bias_init = None

        super().__init__()
        self._build_network()

    def _build_network(self):
        in_features, out_features, num_gaussians = self.hparams.linear[0], self.hparams.linear[-1], self.hparams.num_gaussian
        # self.pi = nn.Linear(self.hparams.input_dim, self.hparams.num_gaussian)
        # nn.init.normal_(self.pi.weight)
        self.sigma = nn.ModuleList([])
        for ind, fc_num in enumerate(self.hparams.linear[:-2]):
            self.sigma.append(nn.Linear(fc_num, self.hparams.linear[ind + 1]))
            # self.sigma.append(nn.BatchNorm1d(flags.linear[ind + 1]))
            self.sigma.append(nn.ELU())
        self.sigma.append(nn.Linear(self.hparams.linear[-2], out_features*num_gaussians))
        self.sigma = nn.Sequential(*self.sigma)

        self.mu = nn.ModuleList([])
        for ind, fc_num in enumerate(self.hparams.linear[:-2]):
            self.mu.append(nn.Linear(fc_num, self.hparams.linear[ind + 1]))
            # self.mu.append(nn.BatchNorm1d(flags.linear[ind + 1]))
            self.mu.append(nn.ELU())
        self.mu.append(nn.Linear(self.hparams.linear[-2], out_features*num_gaussians))
        self.mu = nn.Sequential(*self.mu)
        self.pi = nn.ModuleList([])
        for ind, fc_num in enumerate(self.hparams.linear[:-2]):
            self.pi.append(nn.Linear(fc_num, self.hparams.linear[ind + 1]))
            # self.pi.append(nn.BatchNorm1d(flags.linear[ind + 1]))
            self.pi.append(nn.ELU())
        self.pi.append(nn.Linear(self.hparams.linear[-2], num_gaussians))
        self.pi = nn.Sequential(*self.pi)
        self.out_features, self.num_gaussians = out_features, num_gaussians
        # self.sigma = nn.Linear(
        #     self.hparams.input_dim,
        #     self.hparams.num_gaussian,
        #     bias=self.hparams.sigma_bias_flag,
        # )
        # self.mu = nn.Linear(self.hparams.input_dim, self.hparams.num_gaussian)
        # nn.init.normal_(self.mu.weight)
        # if self.hparams.mu_bias_init is not None:
        #     for i, bias in enumerate(self.hparams.mu_bias_init):
        #         nn.init.constant_(self.mu.bias[i], bias)

    def forward(self, x):
        pi = self.pi(x)
        pi = nn.functional.gumbel_softmax(pi, tau=1, dim=-1) + 1e-15
        sigma = self.sigma(x)
        # Applying modified ELU activation
        sigma = nn.ELU()(sigma) + 1 + 1e-3
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        if torch.any(sigma < 0):
            sigma = sigma.view(-1, 1)
            print('There is sigma smaller than 0!')
            print(sigma[sigma<0])
            quit()
        # print(sigma)
        mu = self.mu(x)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu

    def gaussian_probability(self, sigma, mu, target, log=False):
        """Returns the probability of `target` given MoG parameters `sigma` and `mu`.

        Arguments:
            sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
                size, G is the number of Gaussians, and O is the number of
                dimensions per Gaussian.
            mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
                number of Gaussians, and O is the number of dimensions per Gaussian.
            target (BxI): A batch of target. B is the batch size and I is the number of
                input dimensions.
        Returns:
            probabilities (BxG): The probability of each point in the probability
                of the distribution in the corresponding sigma/mu index.
        """
        target = target.unsqueeze(1).expand_as(mu)
        # target = target.
        if log:
            ret = (
                -torch.log(sigma)
                - 0.5 * LOG2PI
                - 0.5 * torch.pow((target - mu) / sigma, 2)
            )
        else:
            ret = (ONEOVERSQRT2PI / sigma) * torch.exp(
                -0.5 * ((target - mu) / sigma) ** 2
            )
        if torch.any(torch.isnan(ret)):
            print('nan in gaussian probability!!')
            ret = ret.view(-1, 1)
            print(ret[torch.isnan(ret)])
            print(sigma)
            # print(torch.log(sigma))
            # print()
            quit()
        return torch.sum(ret, dim=-1)  # torch.prod(ret, 2)

    def mdn_loss(self, pi, sigma, mu, y):
        log_component_prob = self.gaussian_probability(sigma, mu, y, log=True)
        # print('size of log component', log_component_prob.size())
        log_mix_prob = torch.log(
            pi
            # nn.functional.gumbel_softmax(pi, tau=1, dim=-1) + 1e-15
        )
        # print('size of log mix prob', log_mix_prob.size())
        return -torch.mean(torch.logsumexp(log_component_prob + log_mix_prob, dim=-1))

    def sample(self, pi, sigma, mu):
        """Draw samples from a MoG."""
        categorical = Categorical(pi)
        pis = categorical.sample().unsqueeze(1).unsqueeze(2).expand_as(sigma)
        print('pis size',pis.size())
        # sample = Variable(sigma.data.new(sigma.size(0), ).normal_())
        sample = torch.randn_like(sigma[:,0,:])
        print('sample size', sample.size())
        print('sigma size', sigma.size())
        # sigma.gather()
        print('sigma gather size', sigma.gather(1, pis).size())
        # Gathering from the n Gaussian Distribution based on sampled indices
        sample = sample * sigma.gather(1, pis)[:, 0, :] + mu.gather(1, pis)[:, 0, :]
        return sample

    def generate_samples(self, pi, sigma, mu, n_samples=None):
        if n_samples is None:
            n_samples = self.hparams.n_samples
        samples = []
        softmax_pi = nn.functional.gumbel_softmax(pi, tau=1, dim=-1)
        assert (
            softmax_pi < 0
        ).sum().item() == 0, "pi parameter should not have negative"
        for _ in range(n_samples):
            samples.append(self.sample(softmax_pi, sigma, mu))
        samples = torch.cat(samples, dim=1)
        return samples

    def generate_point_predictions(self, pi, sigma, mu, n_samples=None):
        # Sample using n_samples and take average
        samples = self.generate_samples(pi, sigma, mu, n_samples)
        if self.hparams.central_tendency == "mean":
            y_hat = torch.mean(samples, dim=-1)
        elif self.hparams.central_tendency == "median":
            y_hat = torch.median(samples, dim=-1).values
        return y_hat