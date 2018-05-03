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

print("Loading data set..")
test_raw = genfromtxt('mnist/dataset/mnist_train.csv', delimiter=',')

def segment_data():
    segmented_data = []
    for i in range(10):
        i_examples = (test_raw[test_raw[:,0] == i][:,1:] / 255) - 0.5
        segmented_data.append(i_examples)

    return segmented_data

print("Segmenting...")
segmented_data = segment_data()
print("Dataset loaded!")

class TrainedConvSPN(torch.nn.Module):
    def __init__(self, digits):
        super(TrainedConvSPN, self).__init__()
        self.digits = digits

        # number of examples trained on the last epoch
        self.examples_trained = 0
        self.num_epochs = 0
        self.networks = {}

        self.generate_network()

    def generate_network(self):
        self.shared_parameters = param.Param()

        for digit in self.digits:
            structure = ConvSPN(28, 28, 1, 2)
            structure.print_stat()
            network = MatrixSPN(
                structure,
                self.shared_parameters,
                is_cuda=cuda.is_available())
            self.networks[digit] = network

        self.shared_parameters.register(self)
        self.shared_parameters.proj()

    def save_model(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    def loss_for_digit(self, digit, input):
        network = self.networks[digit]
        (val_dict, cond_mask_dict) = network.get_mapped_input_dict(np.array([input]))
        loss = network.ComputeTMMLoss(val_dict=val_dict, cond_mask_dict=cond_mask_dict)

        return loss

    def compute_total_loss(self, sample_digit, per_network_loss):
        correct_nll= per_network_loss[sample_digit]

        loss = 0
        margin = 10
        for digit in self.digits:
            if digit != sample_digit:
                other_nll = per_network_loss[digit]
                class_loss = (correct_nll - other_nll)
                loss += class_loss

        return loss

    def train_generatively(self, num_sample):
        opt = optim.SGD( self.parameters() , lr=.003)
        self.zero_grad()

        batch = 10
        total_loss = 0

        i = 0
        while i < num_sample:
            self.examples_trained += 1
            for sample_digit in self.digits:
                input = segmented_data[sample_digit][i]

                loss = self.loss_for_digit(sample_digit, input)

                loss.backward()
                total_loss += loss

            if i % batch == 0 or i == num_sample - 1:
                print("Total loss: " + str(i) + " " + str(total_loss[0][0].data))

                #pdb.set_trace()
                if np.isnan(total_loss[0][0].data.cpu().numpy()):
                    return
                total_loss = 0
                opt.step()
                self.zero_grad()
                self.shared_parameters.proj()

            i += 1

    def train_discriminatively(self, num_sample):
        opt = optim.SGD( self.parameters() , lr=.0003)
        self.zero_grad()

        batch = 10
        total_loss = 0

        #pdb.set_trace()

        i = 0
        while i < num_sample:
            self.examples_trained += 1
            for sample_digit in self.digits:
                input = segmented_data[sample_digit][i]

                per_network_loss = {}
                for digit in self.digits:
                    per_network_loss[digit] = self.loss_for_digit(digit, input)

                #print("Sampling " + str(sample_digit) + " with loss " + str(per_network_loss))
                loss = self.compute_total_loss(sample_digit, per_network_loss)

                #pdb.set_trace()
                loss.backward()
                total_loss += loss

            if i % batch == 0 or i == num_sample - 1:
                print("Total loss: " + str(i) + " " + str(total_loss[0][0].data))

                #pdb.set_trace()
                if np.isnan(total_loss[0][0].data.cpu().numpy()):
                    return
                total_loss = 0
                opt.step()
                self.zero_grad()
                self.shared_parameters.proj()

            i += 1

def load_model(filename):
    pass

def main():
    digits_to_train = [7, 8]
    print("Creating SPN")
    tspn = TrainedConvSPN(digits_to_train)
    print("Training SPN")
    tspn.train_generatively(1000)
    tspn.save_model('tmm_gen_' + str(digits_to_train))
    pdb.set_trace()

if __name__ == '__main__':
    main()

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
