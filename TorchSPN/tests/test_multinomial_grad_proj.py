#!/usr/bin/env python3

import numpy as np
import os.path
import sys
from torch import optim
import copy


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import param
from src import network

parameters = param.Param()

net = network.Network()


#######################################
list_prob = [
    np.array([
        [.7, .4],  # associated w/ r.v. X1
        [.2, .3],
        [.1, .3]
    ]),
    np.array([
        [.7, .2],  # associated w/ r.v. X2
        [.3, .8]
    ]),
    np.array([
        [.1, .6],  # associated w/ r.v. X3
        [.9, .4]
    ]),
    np.array([
        [.3, .8],
        [.7, .2]
    ])
]


layer1_multi = net.AddMultinomialNodes(
    n_variable=4,
    n_out=2,
    list_n_values=[3, 2, 2, 2],
    list_prob=list_prob,
    parameters=parameters)
# with size: 6 (= 3 * 2)

#######################################
layer2_prod = net.AddProductNodes(4)

# with size: 3

mask = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [1, 0, 0, 0],
     [0, 1, 0, 0],

     [0, 0, 1, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0],
     [0, 0, 0, 1]],
    dtype='float32')
edge_prod = net.AddProductEdges(layer1_multi, layer2_prod, mask)

#######################################
layer3_sum = net.AddSumNodes(4)


w23 = np.array([
    [.3, .4, 1, 2],
    [.7, .6, 100, 200],
    [12, 2, .9, .1],
    [2,  3, .2, .8]
], dtype='float32')

m23 = np.array([
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 1]
], dtype='float32')


edge23 = net.AddSumEdges(layer2_prod, layer3_sum, w23, m23, parameters=parameters)

#######################################
layer4_prod = net.AddProductNodes(3)

m34 = np.array([
    [1, 0, 0],
    [0, 1, 1],
    [1, 1, 0],
    [0, 0, 1]
], dtype='float32')

prod_edge2 = net.AddProductEdges(layer3_sum, layer4_prod, m34)

#######################################

layer5_sum = net.AddSumNodes(1)


w45 = np.array(
    [[.2],
     [.5],
     [.3]],
    dtype='float32')  # weights always have a shape: lower layer x upper layer

edge_sum = net.AddSumEdges(
    layer4_prod, layer5_sum, w45, parameters=parameters)


parameters.register(net)
################################################
# numeric gradient checking


#################################################
para_vec = copy.copy( parameters.get_unrolled_para() )

# feed value

x_cond_mask = np.array([[0, 0, 0, 0]])

total = 0

opt = optim.SGD( net.parameters(), lr=.03, momentum = 0)


def print_para():
    for p in parameters.para_list:
        print('------')
        print(p)
        print('Hint: look at weights that are reasonable numbers in (0,1)')
        print('They should sum to 1 in each column')

for i in range(300):
    net.zero_grad()
    #if i == 100:
    #    print_para()
    #    pass
    p0000 = net.ComputeProbability({layer1_multi: np.array([[0, 0, 0, 0]])}, {layer1_multi: x_cond_mask}, grad=True)

    if i == 2:
        assert  np.abs(p0000 - 0.07490821927785873 ) < 0.001

    print('p0000 =', float(p0000))

    opt.step()
    parameters.proj()

print("Projected gradient: OK!")
