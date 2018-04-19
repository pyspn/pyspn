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
        [.63],
        [.37]  # associated w/ r.v. X1
    ]),
    np.array([
        [.42],  # associated w/ r.v. X2
        [.58]
    ]),
    np.array([
        [.73],  # associated w/ r.v. X3
        [.27]
    ]),
    np.array([
        [.19],  # associated w/ r.v. X4
        [.81]
    ])
]

multinomial = network_config.AddMultinomialNodes(
    n_variable=4,
    n_out=1,
    list_n_values=[2, 2, 2, 2],
    list_prob=list_prob,
    parameters=parameters)
# with size: 6 (= 3 * 2)

sum_nodes = network_config.AddSumNodes(3)

weights = np.array([
    [.21, 1, 0.33], 
    [1, 0.63, 0.47], 
    [0.79, 0.27, 1],
    [1, 1, 0.53]
],dtype='float32')  # weights always have a shape: lower layer x upper layer

mask = np.array([
    [1, 0, 1], 
    [0, 1, 1], 
    [1, 1, 0],
    [0, 0, 1]
],dtype='float32')

#weights = np.array([
#    [.21, 1, 0, 1], 
#    [1, .63, .27, 1], 
#    [.33, .47, 1, .53]
#],dtype='float32')

edge_sum = network_config.AddSumEdges(
    multinomial, sum_nodes, weights, parameters=parameters, mask=mask)

prod_nodes = network_config.AddProductNodes(1)

mask = np.array([
    [1], 
    [1],
    [1]
],dtype='float32')

#mask = np.array([
#    [1, 1, 1]
#],dtype='float32')

edge_prod = network_config.AddProductEdges(sum_nodes, prod_nodes, mask)

import numpy as np
#################################################
# fee value

x_cond_mask = np.array([
    [0, 0, 0, 0], 
])

#x_cond_mask = np.array([
#    [1, 0, 1, 0], 
#    [0, 1, 1, 0], 
#    [1, 1, 0, 1]
#])

total = 0

res1 = network_config.ComputeUnnormalized({multinomial: np.array([0,0,0,0])})
res2 = network_config.ComputeUnnormalized({multinomial: np.array([1,1,1,1])})

assert np.abs(res1 - xxx) < 0.0001

assert np.abs(res2 - xxx) < 0.0001

print('Good')

#for x1 in range(2):
#    for x2 in range(2):
#        for x3 in range(2):
#            for x4 in range(2):
#                x = np.array([x1, x2, x3, x4])
#                p = network_config.ComputeProbability(
#                    val_dict={multinomial: x},
#                    cond_mask_dict={multinomial: x_cond_mask})
#                total += p
#                print('Prob({}, {}, {}, {}) = {}'.format(x1, x2, x3, x4, float(p)))

#  p(x3=1|x1=0,x2=1)

#x = np.array([[0, 1, 1]])
#x_cond_mask = np.array([[1, 1, 0]])



#p_3_1__1_0_2_1 = network_config.ComputeProbability(
#    val_dict={multinomial: x}, cond_mask_dict={multinomial: x_cond_mask})
#print('p(x3=1|x1=0,x2=1) =', float(p_3_1__1_0_2_1))

#x = np.array([[2, 1, 0]])
#x_cond_mask = np.array([[0, 1, 0]])

# p(x1=2,x2=1|x1=0)
#p_1_2_2_1__1_0 = network_config.ComputeProbability(
#    val_dict={multinomial: x}, cond_mask_dict={multinomial: x_cond_mask})
#print('p(x1=2,x2=1|x1=0) =', float(p_1_2_2_1__1_0))

print('\nSo far so good')


#assert np.abs(p_3_1__1_0_2_1 - 0.565) < 0.001

#assert np.abs(p_1_2_2_1__1_0 - 0.045) < 0.001
