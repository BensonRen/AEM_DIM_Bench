# This is the model maker for the Invertible Neural Network

# From Built-in
from time import time

# From libs
import torch
import torch.nn as nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# From FrEIA
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom


def cINN(flags):
    """
    The constructor for INN network
    :param flags: input flags from configuration
    :return: The INN network
    """
    # Set up the conditional node (y)
    cond_node = ConditionNode(flags.dim_y)
    # Start from input layer
    nodes = [InputNode(flags.dim_x, name='input')]
    # Recursively add the coupling layers and random permutation layer
    for i in range(flags.couple_layer_num):
        nodes.append(Node(nodes[-1], GLOWCouplingBlock,
                          {'subnet_constructor': subnet_fc,
                           'clamp': 2.0},
                          conditions=cond_node,
                          name='coupling_{}'.format(i)))
        nodes.append(Node(nodes[-1], PermuteRandom, {'seed': i}, name='permute_{}'.format(i)))
    # Attach the output Node
    nodes.append(OutputNode(nodes[-1], name='output'))
    nodes.append(cond_node)
    print("The nodes are:", nodes)
    # Return the
    return ReversibleGraphNet(nodes, verbose=True)

##########
# Subnet #
##########
def subnet_fc(c_in, c_out):
    if c_in > 240 and c_in < 280:
        print('INN Chen')
        print('cin = ', c_in)
        return nn.Sequential(nn.Linear(c_in, 160), nn.ReLU(), 
                                  nn.Linear(160, 160), nn.ReLU(),
                                  nn.Linear(160, 160), nn.ReLU(),
                                  nn.Linear(160,  c_out))
    else:
        mid_layer = 1024
        print('INN Peurifoy or Yang')
        print('cin = ', c_in)
    return nn.Sequential(nn.Linear(c_in, mid_layer), nn.ReLU(), 
                        nn.Linear(mid_layer,mid_layer),nn.ReLU(),
                        nn.Linear(mid_layer,  c_out))

# def subnet_fc(c_in, c_out):
#     #Original version of internal layer
#     return nn.Sequential(nn.Linear(c_in, 160), nn.ReLU(), 
#                                   nn.Linear(160, 160), nn.ReLU(),
#                                   nn.Linear(160, 160), nn.ReLU(),
#                                   nn.Linear(160,  c_out))
