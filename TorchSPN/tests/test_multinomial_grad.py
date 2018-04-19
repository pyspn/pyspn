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
        [.1, .6],  # associated w/ r.v. X4
        [.9, .4]
    ]),
    np.array([
        [.2, .5],
        [.8, .5]
    ])
]

multinomial = net.AddMultinomialNodes(
    n_variable=3,
    n_out=2,
    list_n_values=[3, 2, 2],
    list_prob=list_prob,
    parameters=parameters)
# with size: 6 (= 3 * 2)

prod = net.AddProductNodes(3)

# with size: 3

mask = np.array(
    [[1, 1, 0],
     [0, 0, 1],
     [1, 0, 0],
     [0, 1, 1],
     [1, 0, 1],
     [0, 1, 0]],
    dtype='float32')
edge_prod = net.AddProductEdges(multinomial, prod, mask)

sum_final = net.AddSumNodes(1)

weights = np.array(
    [[.2],
     [.5],
     [.3]],
    dtype='float32')  # weights always have a shape: lower layer x upper layer

edge_sum = net.AddSumEdges(
    prod, sum_final, weights, parameters=parameters)


parameters.register(net)
################################################
# numeric gradient checking


#################################################
para_vec = copy.copy( parameters.get_unrolled_para() )

# feed value

x_cond_mask = np.array([[0, 0, 0]])

total = 0

opt = optim.SGD( net.parameters(), lr=.01, momentum = 0)

for i in range(10):
    net.zero_grad()
    p000 = net.ComputeProbability({multinomial: np.array([[0, 0, 0]])}, {multinomial: x_cond_mask}, grad=True)

    ana_grad = parameters.get_unrolled_grad()

    #opt.step()

    print(p000)
#####
# numerical gradient checking

epsilon = 0.001

print(para_vec.shape)

num_grad = []
for i in range( len(para_vec) ):
    if para_vec[i] < epsilon:
        num_grad.append(0.0)
    para_vec[i] += epsilon
    parameters.set_para(para_vec)

    p_plus = -np.log(net.ComputeProbability({multinomial: np.array([[0, 0, 0]])}, {multinomial: x_cond_mask}))

    para_vec[i] -= 2*epsilon
    parameters.set_para(para_vec)

    p_minus = -np.log(net.ComputeProbability({multinomial: np.array([[0, 0, 0]])}, {multinomial: x_cond_mask}))

    para_vec[i] += epsilon

    num_grad.append( float(p_plus - p_minus) / 2/ epsilon)

num_grad = np.array(num_grad).reshape((-1,1))

print('max diff:',np.max(np.abs(ana_grad - num_grad)))
assert( np.max( np.abs( ana_grad - num_grad) ) < 0.001)

print( 'Numerical gradient checking: OK' )
