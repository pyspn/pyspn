import torch
from torch import optim, cuda
import pdb
import math
import csv
import pdb
import numpy as np
import pickle
from numpy import genfromtxt

from collections import defaultdict, deque
from struct_to_spn import *
from timeit import default_timer as timer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from TorchSPN.src import network, param, nodes

class TrainedConvSPN(object):
    def __init__(self, digit):
        self.digit = digit

        # number of examples trained on the last epoch
        self.examples_trained = 0
        self.num_epochs = 0
        self.network = None

        self.generate_network()

    def generate_network(self):
        self.shared_parameters = param.Param()
        structure = ConvSPN(28, 28, 1, 28)
        structure.print_stat()
        self.network = MatrixSPN(
            structure,
            self.shared_parameters,
            is_cuda=cuda.is_available())

        self.shared_parameters.register(self.network)
        self.shared_parameters.proj()

    def classify_data(self, input):
        (val_dict, cond_mask_dict) = self.network.get_mapped_input_dict(np.array([input]))
        prob = self.network.ComputeProbability(
            val_dict=val_dict,
            cond_mask_dict=cond_mask_dict,
            grad=False,
            log=False)

        return prob

    def save_model(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    def train(self, num_sample):
        opt = optim.SGD( self.network.parameters() , lr=.3)
        self.network.zero_grad()

        data = segmented_data[self.digit]

        batch = 1
        total_loss = 0
        for i in range(num_sample):
            sample_index = self.examples_trained

            sample_digit = i % 10
            is_negative = sample_digit != self.digit
            input = segmented_data[sample_digit][sample_index / 10]

            self.examples_trained += 1

            (val_dict, cond_mask_dict) = self.network.get_mapped_input_dict(np.array([input]))
            loss = self.network.ComputeProbability(
                val_dict=val_dict,
                cond_mask_dict=cond_mask_dict,
                grad=True,
                log=True,
                is_negative=is_negative)
            total_loss += loss

            if i % batch == 0 or i == num_sample - 1:
                print("Total loss: " + str(total_loss))
                if np.isnan(total_loss[0][0]):
                    return
                total_loss = 0
                opt.step()
                self.network.zero_grad()
                self.shared_parameters.proj()

def load_model(filename):
    pass

# if __name__ == '__main__':


# print("Loading data set..")
# test_raw = genfromtxt('mnist/dataset/mnist_train.csv', delimiter=',')
#
# def segment_data():
#     segmented_data = []
#     for i in range(10):
#         i_examples = (test_raw[test_raw[:,0] == i][:,1:] / 255) - 0.5
#         segmented_data.append(i_examples)
#
#     return segmented_data
#
# print("Segmenting...")
# segmented_data = segment_data()
# print("Dataset loaded!")

# train_mode = True

# if train_mode:
#     digit_to_train = 3
#     print("Creating SPN")
#     tspn = TrainedConvSPN(digit_to_train)
#     print("Training SPN")
#     tspn.train(85)
#
#     filename = 'neg_spn_3_2'
#     tspn.save_model(filename)

#############################################################

# model5 = pickle.load(open('spn_5', 'rb'))
# model6 = pickle.load(open('spn_6', 'rb'))
#
# def is_data_5(data):
#     is_5 = model5.classify_data(data) > model6.classify_data(data)
#     return is_5
#
# num_tests = 10
# error = 0
# for test_i in range(num_tests):
#     data_5 = segmented_data[5][test_i]
#     data_6 = segmented_data[5][test_i]
#
#     if not is_data_5(data_5):
#         error += 1
#
#     if is_data_5(data_6):
#         error += 1

pdb.set_trace()
