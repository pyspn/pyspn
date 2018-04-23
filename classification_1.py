import numpy as np
import torch
from torch import optim, cuda
import pdb
import math
from collections import defaultdict, deque
from struct_to_spn import *
from timeit import default_timer as timer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from TorchSPN.src import network, param, nodes

x_size = 32
y_size = 32
mspn = MatrixSPN(x_size, y_size, 8, 2, is_cuda=cuda.is_available())

params = mspn.network.parameters()

opt = optim.SGD( params, lr=.003)
mspn.network.zero_grad()

epochs = 10
total_iter = 10

print("SPN generated")
start = timer()
for epoch in range(epochs):
    print("Epoch "+str(epoch))
    for i in range(total_iter):
        fake_input = np.zeros(x_size * y_size)
        mspn.feed(fake_input)
        prob = mspn.forward()
        print(prob)
        opt.step()
        mspn.network.zero_grad()
        mspn.parameters.proj()
end = timer()

print("Done " + str(end - start) + "s")

pdb.set_trace()

'''
Exp1
10 epochs, 10 data
32x32, 8 sum, 2 prd

Dense: 66s
Sparse: 172s

'''
