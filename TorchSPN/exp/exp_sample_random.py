#!/usr/bin/env python3

import numpy as np
import os.path
import sys
from torch import optim

import gl
import load_data

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import param
from src import network


# setup
np.random.seed(1000)

parameters = param.Param()
net = network.Network()


num_leaf = 16

leaves = [None] * num_leaf


def gen_init_param_as(mask):
    weights = np.random.normal(1, 0.2, mask.shape).astype('float32')
    return weights.clip(min=0.5,max=1.5)

for i in range(num_leaf):
    mean = np.random.normal(0,1)
    std  = np.random.normal(1,0.2)
    if std < .6:
        std = .6
    #print('node {}: {}, {}'.format(i, mean, std))

    mean = np.array([mean], dtype='float32')
    std  = np.array([std], dtype='float32')

    leaves[i] = net.AddGaussianNodes(mean, std, parameters=parameters)

layer1_concat = net.AddConcatLayer(leaves)

#####################################################
layer2_prod   = net.AddProductNodes(14)

n_val  = 4
n_dist = 4

mask_1_2 = np.transpose( np.random.multinomial(1, [1.0/n_dist]*n_dist, size=(14, n_val)).reshape(14,-1).astype('float32') )
print(mask_1_2.shape)
net.AddProductEdges(layer1_concat, layer2_prod, mask_1_2)

####################################################
layer3_sum   = net.AddSumNodes(1)

mask_2_3   = np.ones((14,1)).astype('float32')
weight_2_3 = gen_init_param_as(mask_2_3)

net.AddSumEdges(layer2_prod, layer3_sum, weight_2_3, mask_2_3, parameters=parameters)

node2var = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]

####################################################
parameters.register(net)

parameters.proj()

print('Network built!')

#for n in net.nodelist:
#    if len(n.child_edges) == 0:
#        continue
#    print(n.child_edges[0].child)

#train, val, test, node2var = load_data.get_data()


samples = net.GenSample(node2var, n_batch=2)

print(samples)

# write the samples to files as the qu format
exit(1)




mask_dict = {leaves[i]: np.array([[0]]) for i in range(16) }

print('Begin training')

train_loss = 0


