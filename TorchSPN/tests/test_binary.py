#!/usr/bin/env python3

import numpy as np
import os.path
import sys
from torch import optim

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import param
from src import network

parameters = param.Param()

net = network.Network()

input1234 = net.AddBinaryNodes(4)
# with size: 4

x = np.array([[0, 1, 0, 0],
              [1, 0, 1, 1]])


# print(input1234.val)

prod12 = net.AddProductNodes(3)
prod34 = net.AddProductNodes(3)

# with size: 3

mask12 = np.array([[1, 1, 0],
                   [0, 1, 1],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 1],
                   [1, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])

# with size: 8 x 3

edges12 = net.AddProductEdges(input1234, prod12, mask=mask12)


mask34 = np.array([[0, 0, 0],
                   [0, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 1, 1],
                   [1, 0, 1]])

# with size: 8 x 3

edges34 = net.AddProductEdges(input1234, prod34, mask=mask34)

sum12 = net.AddSumNodes(2)
sum34 = net.AddSumNodes(2)

weight12 = np.array([[.2, .1, .7],
                     [.2, .4, .4]], dtype='float32').T

weight34 = np.array([[.4, .1, .5],
                     [.1, .6, .3]], dtype='float32').T


edges12_sum, _ = net.AddSumEdges(
    prod12, sum12, weights=weight12, parameters=parameters)
edges12_sum, _ = net.AddSumEdges(
    prod34, sum34, weights=weight34, parameters=parameters)


prod1234 = net.AddProductNodes(4)

edges_12_1234 = net.AddProductEdges(sum12, prod1234, mask=np.array([[1, 1, 0, 0],
                                                                        [0, 0, 1, 1]]))
edges_12_1234 = net.AddProductEdges(sum34, prod1234, mask=np.array([[1, 0, 1, 0],
                                                                        [0, 1, 0, 1]]))

sum_final = net.AddSumNodes(1)
edges_final = net.AddSumEdges(prod1234, sum_final, weights=np.array(
    [[.2, .5, .1, .2]], dtype='float32').T, parameters=parameters)


#####################################
# compute p_tilde, the input values have already been fed to SPN during
# the construction of SPN
input1234.feed_val(x_onehot=x)

p_tilde = net()
p_tilde = float(p_tilde.data.numpy())

print('p_tilde:', p_tilde)


# compute Z, feed the values at ones
net.feed({input1234: np.ones((2, 4)).astype('float32')})

Z = net()
Z = float(Z.data.numpy())
print('Z      :',  Z)

# print('prob   :',  p_tilde / Z)

print('\nSo far so good for execution (no guarantee for semantics)')
