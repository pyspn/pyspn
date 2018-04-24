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

class ConvSPNArgs(object):
    def __init__(self, x_size, y_size, sum_shifts, prd_subdivs):
        self.x_size = x_size
        self.y_size = y_size
        self.sum_shifts = sum_shifts
        self.prd_subdivs = prd_subdivs

class ClassifierSPN(object):
    def __init__(self, num_classes, model_args, is_cuda=False):
        self.num_classes = num_classes
        self.model_args = model_args
        self.is_cuda = is_cuda

        self.subspns_by_classes = []
        self.shared_parameters = param.Param()
        self.root_sum_weights = []

        self.selected_class_spn = None
        self.selected_class_weight = None
        self.generate_spn()
        self.

    def parameter(self, tensor, requires_grad=True):
        if self.is_cuda:
            tensor = tensor.cuda()
        p = torch.nn.Parameter(tensor, requires_grad=requires_grad)
        shared_parameters.add_param(p, None)

        return p

    def feed(self, class_id, data):
        spn_to_feed = self.subspns_by_classes[class_id]
        spn_to_feed.feed(data)

        self.fed_spn = spn_to_feed

    def forward(self):
        spn_prob = self.selected_class_spn()
        val = self.selected_class_weight * spn_prob
        return val

    def generate_spn(self):
        '''
        1. Create #num_classes subspns
        2. Create #num_classes class leaves
        3. Create product nodes connecting subspns to classes
        4. Create sum nodes
        5. Initialize weights
        ------------------------------
        The above is equivalent to having #num_classes subspns
        '''

        for i in range(self.num_classes):
            # 1. Create subspn and inital weights
            model_args = self.model_args
            new_subspn = MatrixSPN(
                model_args.x_size,
                model_args.y_size,
                model_args.sum_shifts,
                model_args.prd_subdivs,
                self.shared_parameters)

            initial_weight = 1
            new_sum_weight = self.parameter(torch.FloatTensor([initial_weight]))

            # 2. Store it as property for computation later
            self.subspns_by_classes.append(new_subspn)
            self.root_sum_weights.append(new_sum_weight)

    def setup_parameters(self):
        self.shared_parameters.register(self)
        self.shared_parameters.proj()

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
