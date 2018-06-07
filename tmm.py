import torch
from torch import optim, cuda
import pdb
import math
import csv
import pdb
import numpy as np
import pickle
from numpy import genfromtxt
import time
import cProfile

from collections import defaultdict, deque
from struct_to_spn import *
from timeit import default_timer as timer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from TorchSPN.src import network, param, nodes

print("Loading data set..")
test_raw = genfromtxt('train_mnist_16.csv', delimiter=',')

tspn = None

def segment_data():
    segmented_data = []
    for i in range(10):
        i_examples = (test_raw[test_raw[:,0] == i][:,1:] / 255) - 0.5
        segmented_data.append(i_examples)

    min_count = min([arr.shape[0] for arr in segmented_data])

    segmented_tensor = np.zeros((10, min_count, 256))

    for i in range(10):
        segmented_tensor[i] = segmented_data[i][:min_count]

    return segmented_tensor

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
        self.network = None

        self.generate_network()

    def generate_network(self):
        self.shared_parameters = param.Param()

        self.index_by_digits = {}
        for i, digit in enumerate(self.digits):
            self.index_by_digits[digit] = i

        self.structure = MultiChannelConvSPN(16, 16, 1, 2, 1, len(self.digits))
        self.network = MatrixSPN(
            self.structure,
            self.shared_parameters,
            is_cuda=cuda.is_available())

        self.shared_parameters.register(self)
        self.shared_parameters.proj()

    def save_model(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    def compute_total_loss(self, sample_digit, per_network_loss):
        correct_nll= per_network_loss[sample_digit]

        loss = 0
        margin = 1 * len(self.digits)
        for digit in self.digits:
            if digit != sample_digit:
                other_nll = per_network_loss[digit]
                class_loss = (margin + correct_nll - other_nll).clamp(min=0)
                loss += class_loss

        return torch.sum(loss)

    def compute_loss_from_prob_plain(self, sample_digit, prob):
        digit_index = self.index_by_digits[sample_digit]
        return torch.sum(prob[:, digit_index])

    def compute_loss_cross_entropy(self, sample_digit, prob):
        digit_index = self.index_by_digits[sample_digit]
        correct_nll = prob[:, digit_index]
        divisor = torch.sum(prob, 1)

        return torch.sum(correct_nll / divisor)

    def compute_batch_loss_from_prob(self, batch_count_by_digit, prob):
        loss = 0
        batch_start = 0
        for (i, batch_count) in enumerate(batch_count_by_digit):
            digit = self.digits[i]
            batch_end = batch_start + batch_count
            loss += self.compute_loss_from_prob(digit, prob[batch_start:batch_end])

        return loss

    def compute_loss_from_prob(self, sample_digit, prob):
        digit_index = self.index_by_digits[sample_digit]
        correct_nll = prob[:, digit_index]

        loss = 0
        margin = 0 * len(self.digits)
        for digit in self.digits:
            if digit != sample_digit:
                other_nll = prob[:, self.index_by_digits[digit]]
                loss += (margin + correct_nll - other_nll).clamp(min=0)

        return torch.sum(loss)

    def train_generatively(self, num_sample):
        opt = optim.Adam( self.parameters() , lr=.03, weight_decay=0.01)
        self.zero_grad()

        batch = 8
        total_loss = 0

        i = 0
        while i < num_sample:
            self.examples_trained += 1
            for sample_digit in self.digits:
                input = np.tile(segmented_data[sample_digit][i:i+1], 10)

                loss = self.loss_for_digit(sample_digit, input)
                loss.backward()

                total_loss += loss

            if i % batch == 0 or i == num_sample - 1:
                print("Total loss: " + str(i) + " " + str(total_loss[0][0].data))

                if np.isnan(total_loss[0][0].data.cpu().numpy()):
                    return

                total_loss = 0
                opt.step()
                self.zero_grad()
                self.shared_parameters.proj()

            i += 1

    def compute_batch_training_error_from_prob(self, batch_count_by_digit, prob):
        choice = torch.min(prob, 1)[1]
        batch_size = len(prob)

        prob_data = prob.data.cpu().numpy()
        choice = np.argmin(prob_data, 1)

        error = 0
        batch_start = 0
        for i in range(len(self.digits)):
            digit = self.digits[i]
            batch_count = batch_count_by_digit[i]
            batch_end = batch_start + batch_count

            error += torch.sum( choice[batch_start:batch_end] != i )
            batch_end = batch_start

        return error

    def train_discriminatively(self, num_sample_per_digit):
        opt = optim.Adam( self.parameters() , lr=.03)
        pm = list(self.parameters())
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
        #opt = optim.SGD( self.parameters() , lr=.008, momentum=0.9) #, momentum=.0005)
        self.zero_grad()

        batch = 40
        total_loss = 0
        num_sample = num_sample_per_digit * len(self.digits)

        error = 0
        sample_ct = 0
        batch_start_pts = { digit:0 for digit in self.digits }
        while self.examples_trained < num_sample:
            prev_training_count = self.examples_trained

            batch_count_by_digit = []
            input_by_digit = []
            for sample_digit in self.digits:
                data_on_digit = segmented_data[sample_digit]
                num_data_on_digit = data_on_digit.shape[0]
                batch_start = batch_start_pts[sample_digit]
                batch_end = min(batch_start + batch, num_data_on_digit)

                input = np.tile(segmented_data[sample_digit, batch_start:batch_end], self.structure.num_channels)
                input_by_digit.append(input)

                batch_start_pts[sample_digit] = int( batch_end % num_data_on_digit )
                batch_count = batch_end - batch_end + 1
                batch_count_by_digit.append(batch_count)

            input = np.concatenate(input_by_digit)
            self.examples_trained += sum(batch_count_by_digit)

            (val_dict, cond_mask_dict) = self.network.get_mapped_input_dict(np.array([ input ]))
            prob = self.network.ComputeTMMLoss(val_dict=val_dict, cond_mask_dict=cond_mask_dict)

            loss = self.compute_batch_loss_from_prob(batch_count_by_digit, prob)
            loss.backward()

            training_error = self.compute_batch_training_error_from_prob(batch_count_by_digit, prob)

            num_trained_iter = self.examples_trained - prev_training_count
            print("Error: " + str(training_error) +  "\nTotal loss: " + str(self.examples_trained / len(self.digits)) + " " + str(loss[0][0].data / num_trained_iter))

            opt.step()
            self.shared_parameters.proj()
            loss = 0
            self.zero_grad()

def load_model(filename):
    pass

def train_spn():
    print("Training SPN")
    tspn.train_discriminatively(25 * 6000)

start = None
end = None
cprof = None
def cprofile_start():
    global cprof, start
    start = time.time()
    cprof = cProfile.Profile()
    cprof.enable()

def cprofile_end(filename):
    global cprof, start, end
    cprof.disable()
    end = time.time()
    cprof.dump_stats(filename)
    print("Duration: " + str(end - start) + " s")

def main():
    global tspn
    #digits_to_train = [8,9]
    digits_to_train = [0,1,2,3,4,5,6,7,8,9]
    print("Creating SPN")

    tspn = TrainedConvSPN(digits_to_train)
    cprofile_start()
    train_spn()
    cprofile_end("tmm.cprof")
    print("Done")

    tspn.save_model('25ch_x_mmcspn_' + str(digits_to_train).replace(" ", ""))

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

#pdb.set_trace()
