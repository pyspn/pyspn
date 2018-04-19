#!/usr/bin/env python3

import numpy as np
import os.path
import sys
from torch import optim

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import param
from src import network


def Near(a, b):
    return np.abs(a - b) < 0.0001

parameters = param.Param()

net = network.Network()

mean1 = np.array([-.2], dtype='float32')
std1  = np.array([1.3], dtype='float32')

mean2 = np.array([.7], dtype='float32')
std2  = np.array([2.3], dtype='float32')

leaf1 = net.AddGaussianNodes(mean1, std1, parameters=parameters)
leaf2 = net.AddGaussianNodes(mean2, std2, parameters=parameters)

concatlayer = net.AddConcatLayer([leaf1, leaf2])

prod = net.AddProductNodes(1)

net.AddProductEdges(concatlayer,prod)


parameters.register(net)

x1 = np.array([[0.7],
              [-.8]])

x1_cond_mask = np.array([[0.],
                        [0.]])

x2 = np.array([[-0.1],
               [.1]])

x2_cond_mask = np.array([[0.],
                        [0.]])

p = net.ComputeProbability(val_dict={leaf1:x1, leaf2:x2},
                            cond_mask_dict={leaf1:x1_cond_mask, leaf2:x2_cond_mask})

print(p)

assert Near(p[0], 0.039428)
assert Near(p[1], 0.046250)

print('Concatenation layer: OK')