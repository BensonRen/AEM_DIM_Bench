"""A module for a mixture density network layer
This file is copied from https://github.com/search?q=mixture+density+network+pytorch
MIT licensed
For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import math
import numpy as np
#from numpy.random.Generator import multivariate_normal
from torch.distributions.multivariate_normal import MultivariateNormal

ONEOVERSQRT2PI = torch.tensor(1.0 / math.sqrt(2*math.pi), requires_grad=False)

class MDN(nn.Module):
    """A mixture density network layer
    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.
    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions
    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.
    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """
    def __init__(self, flags):
        super(MDN, self).__init__()
        in_features, out_features, num_gaussians = flags.linear[0], flags.linear[-1], flags.num_gaussian
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.ModuleList([])
        self.sigma = nn.ModuleList([])
        self.mu = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear[:-2]):
            self.pi.append(nn.Linear(fc_num, flags.linear[ind + 1]))
            # self.pi.append(nn.BatchNorm1d(flags.linear[ind + 1]))
            self.pi.append(nn.ReLU())
            self.sigma.append(nn.Linear(fc_num, flags.linear[ind + 1]))
            # self.sigma.append(nn.BatchNorm1d(flags.linear[ind + 1]))
            self.sigma.append(nn.ReLU())
            self.mu.append(nn.Linear(fc_num, flags.linear[ind + 1]))
            # self.mu.append(nn.BatchNorm1d(flags.linear[ind + 1]))
            self.mu.append(nn.ReLU())
        self.pi.append(nn.Linear(flags.linear[-2], num_gaussians))
        self.pi.append(nn.Softmax(dim=1))
        # self.sigma.append(nn.Linear(flags.linear[-2], (out_features*(out_features - 1)//2)*num_gaussians))
        self.sigma.append(nn.Linear(flags.linear[-2], (out_features*out_features*num_gaussians)))
        self.mu.append(nn.Linear(flags.linear[-2], out_features*num_gaussians))
        self.pi = nn.Sequential(*self.pi)
        self.mu = nn.Sequential(*self.mu)
        self.sigma = nn.Sequential(*self.sigma)

    def forward(self, minibatch):
        mu = self.mu(minibatch)
        # print('size of mu', mu.size())
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        D = mu.size(-1)

        #print('checki if there is nan in input:', torch.sum(torch.isnan(minibatch)))
        #print('input is: {}'.format(minibatch))
        # print('size of input', minibatch.size())
        pi = self.pi(minibatch)
        # pi = nn.functional.gumbel_softmax(pi, tau=1, dim=-1)
        G = pi.size(1)
        B = pi.size(0)
        # print('D = ', D)
        # print('B = ', B)
        # print('G = ', G)
        # pi = nn.functional.gumbel_softmax(pi, tau=1, dim=-1) + 1e-15
        #print('in forward function, pi = {}'.format(pi))
        # print('self.sigma layer', self.sigma)
        sigma = self.sigma(minibatch)
        # print('size of sigma', sigma.size())
        #sigma = torch.exp(self.sigma(minibatch))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features, self.out_features)
        # diag = torch.diag_embed(torch.diagonal(sigma, dim1=-1, dim2=-2))
        # sigma -= diag
        # sigma += torch.eye(D, device='cuda')#*torch.max(sigma_inv)
        sigma_inv = torch.matmul(sigma.view([-1, D, D]),torch.transpose(sigma.view([-1, D, D]), -1, -2))
        # diag = torch.diag_embed(torch.diagonal(sigma_inv, dim1=-1, dim2=-2))
        # diag = torch.diagonal(sigma_inv, dim1=-1, dim2=-2).unsqueeze(2).expand_as(sigma_inv)
        # print(diag[0])
        # print('size of diag', diag.size())
        # print('size of simga_inv', sigma_inv.size())
        # sigma_inv -= diag
        # sigma_inv += torch.eye(D, device='cuda')*torch.max(sigma_inv)
        # print(sigma_inv[0])
        # print(sigma_inv[1])
        # sigma_inv = torch.div(sigma_inv, diag)
        # sigma_inv = sigma.view(-1, self.num_gaussians, self.out_features, self.out_features)
        # eye = D*torch.eye(D, device='cuda')
        # sigma_inv.add_(eye)
        # sigma_inv = sigma.view(-1, self.out_features, self.out_features)
        det = torch.maximum(torch.det(sigma_inv), torch.tensor(1e-15, device='cuda', requires_grad=False))
        np_det = np.linalg.det(sigma_inv.detach().cpu().numpy()) + 1e-3
        # print('size of det', det.size())
        if torch.any(det < 0):
            negative_index = det < 0
            negative_index = negative_index.detach().cpu().numpy()
            print('index of negative determinants index:', 
                    np.arange(len(det.detach().cpu().numpy()))[negative_index], det[negative_index])
            print('index of NUMPY negative determinant ones', np_det[np_det < 0])
            print('Determinant smaller than zero!!!')
            print('det[det<0]', det[det < 0])
            print('det[negative index]', det[negative_index])
            print('size of det', det.size())
            print('size of simga_inv',sigma_inv.size())
            print('The sigma_inv that has negative index')
            print(sigma_inv[negative_index, :, :])
            print(sigma.view(-1,D,D)[negative_index, :, :])
            print('TORCH re-calculating this det', torch.det(sigma_inv[negative_index, :, :]))
            print('NUMPY re-calculating this det', np.linalg.det(sigma_inv[negative_index, :, :].detach().cpu().numpy()))
            print('now we check the rank of the original matrix and make sure they are 5x5')
            print('rank of the sigma mat is', np.linalg.matrix_rank(sigma.view(-1, D, D)[det<0, :, :].detach().cpu().numpy()))
        sigma_inv = sigma_inv.view(-1, self.num_gaussians, self.out_features, self.out_features)
        # det = torch.det(sigma_inv) + 1e-5
        # # print('after reshape')
        # # print(sigma_inv[0, 0])
        # # print(sigma_inv[0, 1])
        # if torch.any(det < 0):
        #     print('after reshape, Determinant smaller than zero!!!')
        #     print(det[det < 0])
        #     print(sigma_inv[det < 0, :, :])
        # for i in range(10):
        #     print(sigma_inv[i])
        #     print('det of above')
        #     print(torch.det(sigma_inv[i]))
        #print('in forward function, sigma = {}'.format(sigma))
        
        #print('in forward function, mu = {}'.format(mu))
        return pi, sigma_inv, mu


# def gaussian_probability(sigma, mu, target, eps=1e-5):
#     """Returns the probability of `data` given MoG parameters `sigma` and `mu`.
    
#     Arguments:
#         sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
#             size, G is the number of Gaussians, and O is the number of
#             dimensions per Gaussian.
#         mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
#             number of Gaussians, and O is the number of dimensions per Gaussian.
#         data (BxI): A batch of data. B is the batch size and I is the number of
#             input dimensions.
#     Returns:
#         probabilities (BxG): The probability of each point in the probability
#             of the distribution in the corresponding sigma/mu index.
#     """
#     eps = torch.tensor(eps, requires_grad=False)
#     if torch.cuda.is_available():
#         eps = eps.cuda()
#     target = target.unsqueeze(1).expand_as(mu)
#     #ll = torch.log(ONEOVERSQRT2PI / sigma) * (-0.5 * ((target - mu) / sigma)**2)
#     #return torch.sum(ll, 2)
#     ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / (sigma + eps)
#     #ret = 0.5*torch.matmul((target - mu)
#     return torch.max(torch.prod(ret, 2), eps)

def mdn_loss(pi, sigma_inv, mu, target):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    #GP = gaussian_probability(sigma, mu, target)
    D = mu.size(-1)
    target = target.unsqueeze(1).expand_as(mu)
    loss = 0
    G = pi.size(1)
    B = pi.size(0)
    
    #########################
    # matrix implementation #
    #########################
    diff = target - mu
    diff = diff.view(-1, D)
    # print('size of diff = ', diff.size())
    #print('size of sigma = ', sigma.size())
    # To make sigma symmetric add the transpose to itself
    #precision_mat_diag_pos = torch.matmul(sigma.view([-1, D, D]),torch.transpose(sigma.view([-1, D, D]), 1, 2))
    #print('size of precision mat is', precision_mat_diag_pos.size())
    #precision_mat =  sigma.view([-1, D, D]) + torch.transpose(sigma.view([-1, D, D]), 1, 2)
    #diagonal_mat = torch.zeros(D,D).cuda()
    #precision_mat_diag_pos = precision_mat + diagonal_mat.fill_diagonal_(1 - torch.min(torch.diagonal(precision_mat)).detach().cpu().numpy())
    
    # mul1 = torch.transpose(torch.diagonal(torch.matmul(diff.view([-1, D]), sigma)),0, 1)
    #print('size of mul1 = ', mul1.size())
    #diff_t =  torch.transpose(diff.view([-1, D]),0, 1)
    # diff_t = torch.transpose(diff, 0, 1)
    # print('size of diff_t = ', diff_t.size())
    sigma_inv = sigma_inv.view(-1, D, D)
    det = torch.maximum(torch.det(sigma_inv), torch.tensor(1e-15, device='cuda', requires_grad=False))
    # print('size of sigma inv is', sigma_inv.size())
    left_mul = torch.matmul(diff.unsqueeze(1), sigma_inv)
    # print('size of left mul', left_mul.size())
    mul = torch.matmul(left_mul, diff.unsqueeze(2))
    # print('mul size = ', mul.size())
    mul = mul.view(B, G)
    # print('mul size = ', mul.size())
    p_x_right = -0.5*torch.square(mul)
    # print('size of px_right', p_x_right.size())
    sigma_inv = sigma_inv.view(B, G, D, D)
    # det = torch.det(sigma_inv)
    det = det.view(B, G)
    p_x_left = torch.log(ONEOVERSQRT2PI) + torch.log(torch.sqrt(det))
    # print('size of px_left', p_x_left.size())
    # print('size of p1', pi.size())
    ret =  -torch.logsumexp(torch.log(pi) + p_x_left + p_x_right, dim=-1)
    # print(ret)
    # print('p_x_right', p_x_right[0])
    # print('det', det[0])
    # print('p_x_left', p_x_left[0])
    # # print('sigma_inv', sigma_inv)
    # print('left_mult',left_mul[0])
    # print('mul',mul[0])
    # print('pi', pi[0])
    # print('log pi', torch.log(pi)[0])
    if torch.any(torch.isnan(ret)) or torch.any(torch.isnan(p_x_left)) or torch.any(torch.isnan(p_x_right)):
        det = det.view(-1)
        sigma_inv = sigma_inv.view(-1, D, D)
        negative_index = det < 0
        negative_index = negative_index.detach().cpu().numpy()
        print('index of negative determinants index:', negative_index)
        print('index:', np.arange(len(det.detach().cpu().numpy()))[negative_index])
        print('the determinants are', det[negative_index])
        # print('index of NUMPY negative determinant ones', np_det[np_det < 0])
        print('Determinant smaller than zero!!!')
        print('det[det<0]', det[det < 0])
        print('det[negative index]', det[negative_index])
        print('size of det', det.size())
        print('size of simga_inv',sigma_inv.size())
        print('The sigma_inv that has negative index')
        print(sigma_inv[negative_index, :, :])
        # print(sigma.view(-1,D,D)[negative_index, :, :])
        print('TORCH re-calculating this det', torch.det(sigma_inv[negative_index, :, :]))
        print('NUMPY re-calculating this det', np.linalg.det(sigma_inv[negative_index, :, :].detach().cpu().numpy()))
        print('now we check the rank of the original matrix and make sure they are 5x5')
        # print('rank of the sigma mat is', np.linalg.matrix_rank(sigma.view(-1, D, D)[det<0, :, :].detach().cpu().numpy()))
        print('ret has nan: ',torch.any(torch.isnan(ret)))
        print('p_x_left has nan: ',torch.any(torch.isnan(p_x_left)))
        print('p_x_right has nan: ',torch.any(torch.isnan(p_x_right)))
        print('p_x_right', p_x_right)
        print('p_x_left', p_x_left)
        print('Test the part of p_x_left ', p_x_left[torch.isnan(p_x_left)])
        print('The det that is giving trouble of these p_x_left', det.view(B, G)[torch.isnan(p_x_left)])
        print('The original sigma_inv is ', sigma_inv.view(B,G,D,D)[torch.isnan(p_x_left),:,:])
        print('det', det)
        # print('sigma_inv', sigma_inv)
        print('left_mult',left_mul)
        print('mul',mul)
        print('pi', pi)
        quit()
    # print('size of return', ret.size())
    return torch.mean(ret)
    quit()
    p_value =  torch.diagonal(torch.matmul(mul1,diff_t)).view([B, G])
    #print('size of p_value = ', p_value.size())
    #print('p_value', p_value)
    det_sigma = torch.abs(torch.det(sigma)).view([B,G])
    #det_sigma = torch.det(precision_mat_diag_pos).view([B,G])
    #print('deg_sigma', det_sigma)
    # before_exp = torch.minimum(torch.log(pi) + 0.5*torch.log(det_sigma) - 0.5*p_value, 
    #                       other=torch.tensor(50.,requires_grad=False).cuda())     # capping to e^50
    #print('before_exp', before_exp)
    likelihood = torch.exp(before_exp)
    #print('likihood' , likelihood)
    #loss = torch.mean(-torch.log(torch.sum(likelihood, dim=1)))
    loss = torch.mean(-torch.log(torch.sum(likelihood, dim=1)+1e-6))
    #print('loss = ', loss)
    return loss
    """
    loss = torch.sum(0.5*p_value, dim=1)
    #print('size of det sigma', det_sigma.size())
    #print(det_sigma)
    sigma_term = -torch.sum(torch.log(pi*torch.sqrt(det_sigma.view([B, G]))), dim=1)
    #print('size of sigma term = ', sigma_term.size())
    #print('sigma term', sigma_term)
    loss += sigma_term
    mean_loss = torch.mean(loss)
    #print(mean_loss)
    #exit()
    return mean_loss
    #loss = torch.sum(torch.matmul(pi, p_value)
    """

    """
    ##########################
    # individual computation #
    ##########################
    for g in range(G):
        for b in range(B):
            diff = target[b, g, :] - mu[b, g, :]
            #print(diff.size())
            #print(sigma[:,g,:,:].size())
            p_value =  torch.matmul(torch.matmul(diff, sigma[b,g,:,:]),diff)# torch.transpose(diff))
            loss +=  0.5 * pi[b, g] * p_value                 # The diagonal part 
        loss += -torch.matmul(pi[:,g], torch.log(torch.sqrt(torch.det(sigma[:,g,:,:]))))
    print(loss.size())
    return loss
    #prob = pi*GP
    #prob = torch.log(pi)+ GP
    #print('pi part: {}, gaussian_part: {}'.format(pi, GP))
    #print('prob size = {}'.format(prob.size()))
    #for i in range(prob.size(1)):
    #    for j in range(prob.size(0)):
    #        print('prob {} = {}'.format(i, prob[j, i]))
    #print('sum(exp(prob)) = {}'.format(torch.sum(prob, dim=1)))
    #print('-log (sum(exp(prob)))={}'.format(-torch.log(torch.sum(prob, dim=1))))
    #print('mean = {}'.format(torch.mean(-torch.log(torch.sum(prob, dim=1)))))
    
    #nll =  -torch.log(torch.sum(prob, dim=1))
    #nll = -torch.sum(prob, dim=1)
    #nll = nll[torch.logical_not(torch.isinf(nll))]
    #nll = nll[torch.logical_not(torch.isnan(nll))]
    #print('mean = {}'.format(torch.mean(nll)))
    #return torch.mean(nll)
    """

def sample(pi, sigma, mu):
    """Draw samples from a MoG.
    # Original implementation
    categorical = Categorical(pi)
    pis = list(categorical.sample().data)
    sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_())
    for i, idx in enumerate(pis):
        sample[i] = sample[i].mul(sigma[i,idx]).add(mu[i,idx])
    return sample
    """
    ######################
    # new implementation #
    ######################
    categorical = Categorical(pi)
    pis = list(categorical.sample().data)
    #print('len of pis', len(pis))
    #print('pis', pis)
    print('size of sigma = ', sigma.size())
    print('size of mu = ', mu.size())
    D = mu.size(-1)
    samples = torch.zeros([len(pi), D])
    sigma_cpu_all = sigma.detach().cpu()
    mu_cpu_all = mu.detach().cpu()
    for i, idx in enumerate(pis):
        #print('i = {}'.format(i))
        sigma_cpu = sigma_cpu_all[i,idx]
        precision_mat_diag_pos = torch.matmul(sigma_cpu,torch.transpose(sigma_cpu,0,1))
        mu_cpu = mu_cpu_all[i, idx]
        #precision_mat = sigma[i, idx] + torch.transpose(sigma[i, idx], 0, 1)
        diagonal_mat = torch.tensor(np.zeros([D,D]))
        #precision_mat_diag_pos np.fill_diagonal_(diagonal_mat, 1e-7)
        precision_mat_diag_pos += diagonal_mat.fill_diagonal_(1)    # add small positive value
        #precision_mat_diag_pos = precision_mat + diagonal_mat.fill_diagonal_(1 - torch.min(torch.diagonal(precision_mat)).detach().cpu().numpy())
        #print('precision_mat = ', precision_mat_diag_pos)
        #print(precision_mat_diag_pos)
        #print(mu_cpu)
        try:
            #print('precision_mat = ', precision_mat_diag_pos)
            MVN = MultivariateNormal(loc=mu_cpu, precision_matrix=precision_mat_diag_pos)
            draw_sample = MVN.rsample()
        except:
            print("Ops, your covariance matrix is very unfortunately singular, assign loss of test_loss to avoid counting")
            draw_sample = -999*torch.ones([1, D])
        #print('sample size = ', draw_sample.size())
        samples[i, :] = draw_sample
    #print('samples', samples.size())
    return samples
def new_mdn_loss(pi, sigma, mu, target):
    """
    Copied from :
    https://github.com/sksq96/pytorch-mdn/blob/master/mdn.ipynb
    """
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    loss = torch.exp(m.log_prob(target))
    loss = torch.sum(loss * pi, dim=1)
    loss = -torch.log(loss)
    return torch.mean(loss)