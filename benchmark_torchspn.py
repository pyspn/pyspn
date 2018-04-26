import numpy as np
import torch
from torch import optim, cuda
import pdb
import math
import pickle
from collections import defaultdict, deque
from struct_to_spn import *
from timeit import default_timer as timer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from TorchSPN.src import network, param, nodes

shared_parameters = param.Param()

x_size = 32
y_size = 32
# cspn = ConvSPN(x_size, y_size, 8, 2)
cspn = FlatSPN(x_size, y_size)
mspn = MatrixSPN(cspn, shared_parameters, is_cuda=cuda.is_available())

shared_parameters.register(mspn)
shared_parameters.proj()

params = mspn.parameters()

opt = optim.SGD( params, lr=.003)
mspn.zero_grad()

epochs = 10
total_iter = 10

print("SPN generated")

filename = 'small.spn'
start = timer()
for epoch in range(epochs):
    print("Epoch "+str(epoch))
    for i in range(total_iter):
        fake_input = np.zeros(x_size * y_size)
        (val_dict, cond_mask_dict) = mspn.get_mapped_input_dict(np.array([fake_input]))
        loss = mspn.ComputeProbability(
            val_dict=val_dict,
            cond_mask_dict=cond_mask_dict,
            grad=True,
            log=True)

        opt.step()
        mspn.zero_grad()
        shared_parameters.proj()

end = timer()

print("Done " + str(end - start) + "s")

'''
Exp1
10 epochs, 10 data
32x32, 8 sum, 2 prd

Dense: 66s
Sparse: 172s
'''


'''
Define nodes as layers, connect nodes

Multiply by 1/|unobserved|
'''
