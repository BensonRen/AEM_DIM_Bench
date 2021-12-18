"""
This is the module where the model is defined. It uses the nn.Module as backbone to create the network structure
"""
# Own modules

# Built in
import math
# Libs
import numpy as np

# Pytorch module
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import pow, add, mul, div, sqrt
import mdn


class MDN(nn.Module):
    def __init__(self, flags):
        super(MDN, self).__init__()
        # Linear Layer and Batch_norm Layer definitions here
        # self.linears = nn.ModuleList([])
        # self.bn_linears = nn.ModuleList([])
        # for ind, fc_num in enumerate(flags.linear[0:-2]):               # Excluding the last one as we need intervals
        #     self.linears.append(nn.Linear(fc_num, flags.linear[ind + 1]))
        #     self.bn_linears.append(nn.BatchNorm1d(flags.linear[ind + 1]))

        # The mixture density network uses 3 hyper-parameters for initialization (#in_features, #out_features, #Guassians)
        self.mdn = mdn.MDN(flags)

    def forward(self, G):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The 300 dimension spectra
        """
        out = G                                                         # initialize the out
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            if ind != len(self.linears) - 1:
                out = F.relu(bn(fc(out)))                                   # ReLU + BN + Linear
                #out = F.relu(fc(out))                                   # ReLU + BN + Linear
            else:
                out = F.relu(fc(out))

        # The mixture density network outputs 3 values (Pi is a multinomial distribution of the Gaussians. Sigma
        #             is the standard deviation of each Gaussian. Mu is the mean of each
        #             Gaussian.)
        pi, sigma, mu = self.mdn(out)
        return pi, sigma, mu
