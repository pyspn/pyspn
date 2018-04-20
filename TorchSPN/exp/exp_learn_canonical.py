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
np.random.seed(123)

parameters = param.Param()
net = network.Network()


num_leaf = 16

leaves = [None] * num_leaf

def gen_complete_binary_tree_mask(n):
    # the mask will be np.array
    # of shape 2n x n
    mask = np.eye(n, dtype='float32').reshape(n, 1, n)
    mask = np.tile(mask, (1, 2, 1)).reshape(2*n, n)
    return mask

def gen_init_param_as(mask):
    weights = np.random.normal(1, 0.2, mask.shape).astype('float32')
    return weights.clip(min=0.5,max=1.5)

for i in range(num_leaf):
    mean = np.random.normal(0,1)
    std  = np.random.normal(1,0.2)
    if std < .6:
        std = .6
    mean = np.array([mean], dtype='float32')
    std  = np.array([std], dtype='float32')

    leaves[i] = net.AddGaussianNodes(mean, std, parameters=parameters)

layer1_concat = net.AddConcatLayer(leaves)

#####################################################
layer2_prod   = net.AddProductNodes(8)

mask_1_2 = gen_complete_binary_tree_mask(layer2_prod.num)

net.AddProductEdges(layer1_concat, layer2_prod, mask_1_2)

####################################################
layer3_sum   = net.AddSumNodes(4)

mask_2_3   = gen_complete_binary_tree_mask(layer3_sum.num)
weight_2_3 = gen_init_param_as(mask_2_3)

net.AddSumEdges(layer2_prod, layer3_sum, weight_2_3, mask_2_3, parameters=parameters)

####################################################
layer4_prod = net.AddProductNodes(2)

mask_3_4 = gen_complete_binary_tree_mask(2)
net.AddProductEdges(layer3_sum, layer4_prod, mask_3_4)

####################################################
layer5_sum = net.AddSumNodes(1)

mask_4_5   = gen_complete_binary_tree_mask(1)
weight_4_5 = gen_init_param_as(mask_4_5)

net.AddSumEdges(layer4_prod, layer5_sum, weight_4_5, mask_4_5, parameters=parameters)

parameters.register(net)

parameters.proj()

print('Network built!')

node2var = [0,1,0,1,2,3,2,3,0,1,0,1,2,3,2,3]
train, val, test = load_data.get_data('QU.txt', node2var)

print('Data:')
print('Train: ', train.shape)
print('CV:    ', val.shape)
print('Test:  ', test.shape)

mask_dict = {leaves[i]: np.array([[0]]) for i in range(16) }

batch = 10

sample_cnt = 0

opt = optim.SGD( net.parameters(), lr=.003/batch, momentum = 0.9)
net.zero_grad()

print('Begin training')

train_loss = 0


for epoch in range(1000):

    for x in train:
        #if gl.debug == True:
        #    gl.debug = False
        #    gl.debugged = True
        #if epoch == 1 and not gl.debugged:
        #    gl.debug = True
        #if sample_cnt > 528:
        #    print(sample_cnt, x)
        #    gl.debug = True
        #    for idx, p in enumerate(parameters.para_list):
        #        print(idx , p.data.numpy())

        val_dict = {leaves[i]:np.array([[ x[i] ]]) for i in range(len(x))}
        train_loss += net.ComputeProbability(val_dict=val_dict, cond_mask_dict=mask_dict, grad=True, log=True)

        sample_cnt += 1


        if sample_cnt % batch == 0:
            opt.step()
            net.zero_grad()
            parameters.proj()

        if sample_cnt != 1 and sample_cnt % 100 == 0:

            #for idx, p in enumerate(parameters.para_list):
            #    print(idx, p.data.numpy())

            #gl.debug = True


            val_loss = 0
            for x in val:
                val_dict = {leaves[i]: np.array([[x[i]]]) for i in range(len(x))}
                logp = net.ComputeProbability(val_dict=val_dict, cond_mask_dict=mask_dict, grad=False, log=True)
                val_loss += logp
            print('Epoch {}, Train loss {}, Val loss {}'.format(epoch, train_loss/100., val_loss/len(val)))
            train_loss = 0
