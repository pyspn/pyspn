#!/usr/bin/env python3

import numpy as np
import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import param
from src import network

parameters = param.Param()

network_config = network.Network()

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
    ])
]

multinomial = network_config.AddMultinomialNodes(
    n_variable=3,
    n_out=2,
    list_n_values=[3, 2, 2],
    list_prob=list_prob,
    parameters=parameters)
# with size: 6 (= 3 * 2)

prod = network_config.AddProductNodes(3)
# with size: 3

mask = np.array(
    [[1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0]],
    dtype='float32')
edge_prod = network_config.AddProductEdges(multinomial, prod, mask)

sum_final = network_config.AddSumNodes(1)

weights = np.array(
    [[.2], [.5], [.3]],
    dtype='float32')  # weights always have a shape: lower layer x upper layer

edge_sum = network_config.AddSumEdges(
    prod, sum_final, weights, parameters=parameters)

#################################################
# fee value

x = np.array([[0,0,0],
              [0,0,1],
              [0,1,0],
              [0,1,1],
              [1,0,0],
              [1,0,1],
              [1,1,0],
              [1,1,1]])

x_cond_mask = np.array([[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]])

p = network_config.ComputeProbability(
                val_dict={multinomial: x},
                cond_mask_dict={multinomial: x_cond_mask})

print(p)


x = np.array([[0, 1, 1],
              [2, 1, 0]])

m = np.array([[1, 1, 0],
              [0, 1, 0]])

p = network_config.ComputeProbability(
                val_dict={multinomial: x},
                cond_mask_dict={multinomial: m})


print(p)

p_3_1__1_0_2_1 = p[0]
p_1_2_2_1__1_0 = p[1]

assert np.abs(p_3_1__1_0_2_1 - 0.565) < 0.001

assert np.abs(p_1_2_2_1__1_0 - 0.045) < 0.001

print('Batch implementation: OK')

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