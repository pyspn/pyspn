import torch
import pdb
from torch import optim, cuda
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
import data_loader

(train_data, test_data) = data_loader.load_data(data_loader.mnist_16_filename)

class Hyperparameter(object):
    def __init__(self, structure=None, optimizer_constructor=None, loss=None, batch_size=None):
        self.structure = structure
        self.optimizer_constructor = optimizer_constructor
        self.loss = loss
        self.batch_size = batch_size

class TrainingStatistics(object):
    def __init__(self, filename='default.perf'):
        self.data = []
        self.filename = filename
        self.examples_trained = 0

class TrainedConvSPN(torch.nn.Module):
    def __init__(self, digits=None, hyperparameter=None, use_cuda=cuda.is_available()):
        super(TrainedConvSPN, self).__init__()
        self.digits = digits

        self.use_cuda = use_cuda

        self.hyperparameter = hyperparameter

        self.stats = TrainingStatistics()

        self.network = None
        self.shared_parameters = None
        self.optimizer = None
        self.generate_network()

    def generate_network(self):
        self.shared_parameters = param.Param()

        self.index_by_digits = {}
        for i, digit in enumerate(self.digits):
            self.index_by_digits[digit] = i

        self.network = MatrixSPN(
            self.hyperparameter.structure,
            self.shared_parameters,
            is_cuda=self.use_cuda)

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
        return torch.sum(-prob[:, digit_index])

    def compute_batch_cross_entropy(self, batch_count_by_digit, prob):
        target = torch.LongTensor( np.repeat( np.arange(len(batch_count_by_digit)), batch_count_by_digit) )

        if self.use_cuda:
            target = target.cuda()

        target = torch.autograd.Variable( target )

        loss_fn = torch.nn.CrossEntropyLoss()

        return loss_fn( -prob, target )

    def compute_batch_loss_from_prob(self, batch_count_by_digit, prob):
        return self.compute_batch_hinge_loss( batch_count_by_digit, prob )
        #return self.compute_batch_cross_entropy( batch_count_by_digit, prob )

    def compute_batch_hinge_loss(self, batch_count_by_digit, prob):
        loss = 0
        batch_start = 0
        for (i, batch_count) in enumerate(batch_count_by_digit):
            digit = self.digits[i]
            batch_end = batch_start + batch_count
            loss += self.compute_loss_from_prob(digit, prob[batch_start:batch_end])
            batch_start = batch_end

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
            self.stats.examples_trained += 1
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

            error += np.sum( choice[batch_start:batch_end] != i )
            batch_start = batch_end

        return error/batch_size

    def validate_network(self):
        validation_size = 20
        batch_count_by_digit = []
        input_by_digit = []
        for sample_digit in self.digits:
            data_on_digit = train_data[sample_digit]
            num_data_on_digit = data_on_digit.shape[0]

            input_i = np.tile(test_data[sample_digit, 0:validation_size], self.hyperparameter.structure.num_channels)
            input_by_digit.append(input_i)

            batch_count_by_digit.append(validation_size)

        input_batch = np.concatenate(input_by_digit)

        (val_dict, cond_mask_dict) = self.network.get_mapped_input_dict(np.array([ input_batch ]))
        prob = self.network.ComputeTMMLoss(val_dict=val_dict, cond_mask_dict=cond_mask_dict, debug=True)

        loss = self.compute_batch_loss_from_prob(batch_count_by_digit, prob).data.cpu().numpy() / (validation_size * len(self.digits))
        training_error = self.compute_batch_training_error_from_prob(batch_count_by_digit, prob)

        res = (float(training_error), float(loss))

        del prob, loss, training_error

        return res

    def train_discriminatively(self, num_sample_per_digit):
        self.optimizer = self.hyperparameter.optimizer_constructor( self.parameters() )

        self.zero_grad()

        batch = self.hyperparameter.batch_size

        total_loss = 0
        num_sample = num_sample_per_digit * len(self.digits)

        error = 0
        sample_ct = 0
        last_print = 0
        last_validation = 0

        batch_start_pts = { digit:0 for digit in self.digits }
        while self.stats.examples_trained < num_sample:
            prev_training_count = self.stats.examples_trained

            batch_count_by_digit = []
            input_by_digit = []
            for sample_digit in self.digits:
                data_on_digit = train_data[sample_digit]
                num_data_on_digit = data_on_digit.shape[0]
                batch_start = batch_start_pts[sample_digit]
                batch_end = min(batch_start + batch, num_data_on_digit)

                input_i = np.tile(train_data[sample_digit, batch_start:batch_end], self.hyperparameter.structure.num_channels)
                input_by_digit.append(input_i)

                batch_start_pts[sample_digit] = int( batch_end % num_data_on_digit )
                batch_count = batch_end - batch_start

                batch_count_by_digit.append(batch_count)

            input_batch = np.concatenate(input_by_digit)
            self.stats.examples_trained += sum(batch_count_by_digit)

            (val_dict, cond_mask_dict) = self.network.get_mapped_input_dict(np.array([ input_batch ]))
            prob = self.network.ComputeTMMLoss(val_dict=val_dict, cond_mask_dict=cond_mask_dict)

            loss = self.compute_batch_loss_from_prob(batch_count_by_digit, prob)
            loss.backward()

            training_error = self.compute_batch_training_error_from_prob(batch_count_by_digit, prob)
            num_trained_iter = self.stats.examples_trained - prev_training_count

            self.optimizer.step()
            self.shared_parameters.proj()

            if self.stats.examples_trained - last_print > 10000:
                last_print = self.stats.examples_trained

                training_loss = loss[0][0].data.cpu().numpy() / num_trained_iter

                if self.stats.examples_trained - last_validation > 60000:
                    last_validation = self.stats.examples_trained
                    print("Validating network " + str(int(self.stats.examples_trained / 60000)))
                    (validation_error, validation_loss) = self.validate_network()

                    pd = [float(training_error), float(training_loss), float(validation_error), float(validation_loss)]
                    self.stats.data.append(pd)

                    np.savetxt(self.stats.filename, np.array(self.stats.data))
                    print("Validation Error: " + str(validation_error) + "\nValidation Loss: " + str(validation_loss))

                print("Training Error: " + str(training_error) +  "\nTraining loss: " + str(self.stats.examples_trained / len(self.digits)) + " " + str(training_loss))

            loss = 0
            self.zero_grad()

tspn = None

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

    digits_to_train = [0,1,2,3,4,5,6,7,8,9]
    structure = MultiChannelConvSPN(16, 16, 1, 2, 40, len(digits_to_train))
    hyperparameter = Hyperparameter(
            structure=structure,
            optimizer_constructor=(lambda param: torch.optim.Adam(param, lr=0.05)),
            batch_size=32)

    print("Creating SPN")

    tspn = TrainedConvSPN(digits=digits_to_train, hyperparameter=hyperparameter)

    cprofile_start()
    train_spn()
    cprofile_end("tmm.cprof")
    print("Done")

    tspn.save_model('plain_' + str(digits_to_train).replace(" ", ""))

    pdb.set_trace()

def retrain(model_name):
    model = pickle.load(open(model_name, 'rb'))

    print("Retraining " + model_name)
    pdb.set_trace()


if __name__ == '__main__':
    main()
    #retrain('m3')
