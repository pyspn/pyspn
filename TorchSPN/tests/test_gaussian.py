#!/usr/bin/env python3

import numpy as np
import os.path
import sys
from torch import optim, cuda

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import param
from src import network


def Near(a, b):
    return np.abs(a - b) < 0.0001

parameters = param.Param()

net = network.Network(is_cuda=cuda.is_available())

mean = np.array([-.2], dtype='float32')
std  = np.array([1.3], dtype='float32')

leaf1 = net.AddGaussianNodes(mean, std, parameters=parameters)
parameters.register(net)

x = np.array([[0.7],
              [-.8]])

x_cond_mask = np.array([[0.],
                        [0.]])

p0 = net.ComputeProbability(val_dict={leaf1:x},
                            cond_mask_dict={leaf1:x_cond_mask})

print(p0)

assert np.abs( p0[0] - 0.2414850 ) < 0.0001
assert np.abs( p0[1] - 0.2758738 ) < 0.0001

print("Forward pass for Gaussian nodes: OK\n")


opt = optim.SGD( net.parameters(), lr=.01, momentum = 0)

x = np.array([[0.9],
              [0.7]])


for i in range(400):
    net.zero_grad()

    p = net.ComputeProbability(val_dict={leaf1:x},
                            cond_mask_dict={leaf1:x_cond_mask}, grad=True, log=True)

    opt.step()
    parameters.proj()

print(p)

mu    = leaf1.mean.data.cpu().numpy().__float__()

sigma = np.exp(leaf1.logstd.data.cpu().numpy().__float__())

print(mu, sigma)

assert Near(mu, 0.8)
assert Near(sigma, 0.1)

print("Fitting empirical Gaussian: OK")


'''
Prob(0, 0, 0) = 0.05420000106096268
Prob(0, 0, 1) = 0.13779999315738678
Prob(0, 1, 0) = 0.1818000078201294
Prob(0, 1, 1) = 0.2362000048160553
Prob(1, 0, 0) = 0.016600001603364944
Prob(1, 0, 1) = 0.04939999803900719
Prob(1, 1, 0) = 0.05640000104904175
Prob(1, 1, 1) = 0.10760000348091125
Prob(2, 0, 0) = 0.009199999272823334
Prob(2, 0, 1) = 0.03279999643564224
Prob(2, 1, 0) = 0.03180000185966492
Prob(2, 1, 1) = 0.08619999885559082
p(x3=1|x1=0,x2=1) = 0.5650717616081238
p(x1=2,x2=1|x1=0) = 0.045428574085235596
'''
