#!/usr/bin/env python3

import numpy as np
import os.path
import sys
from torch import optim, cuda

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import param
from src import network

parameters = param.Param()

def create_network1():
    net = network.Network(is_cuda=cuda.is_available())

    leaves = net.AddBinaryNodes(2)

    sum1 = net.AddSumNodes(3)

    weights1 = np.array([[2,8,0,0],
                        [1,9,0,0],
                        [0,0,4,6]], dtype='float32').T
    edges1 = net.AddSumEdges(leaves, sum1, weights=weights1, parameters=parameters)

    prod1 = net.AddProductNodes(2)
    mask1 = np.array([[1,0],
                      [0,1],
                      [1,1]])
    edges2 = net.AddProductEdges(sum1, prod1, mask=mask1)

    sum_final = net.AddSumNodes(1)
    weights_final = np.array([[.3, .7]], dtype='float32').T
    edges_final = net.AddSumEdges(prod1, sum_final, weights=weights_final,
                                    parameters=parameters)

    return (leaves, net)

(leaves, net) = create_network1()

p = net.ComputeUnnormalized({leaves: np.array([[0,1],[1,0]], dtype='float32')})
Z = net.ComputeUnnormalized({leaves: np.ones((2, 2)).astype('float32')})

print('p:', p)
print('Z:', Z)
