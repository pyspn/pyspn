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
from cifar10.dataset.cifar_extractor import *
from TorchSPN.src import network, param, nodes

segmented_data = None

class TrainedConvSPN(torch.nn.Module):
    def __init__(self, img_keys):
        super(TrainedConvSPN, self).__init__()
        self.img_keys = img_keys

        # number of examples trained on the last epoch
        self.examples_trained = 0
        self.num_epochs = 0
        self.networks = {}

        self.generate_network()

    def generate_network(self):
        self.shared_parameters = param.Param()

        for img_key in self.img_keys:
            structure = MultiChannelConvSPN(32, 32, 1, 2, 3)
            structure.print_stat()
            network = MatrixSPN(
                structure,
                self.shared_parameters,
                is_cuda=cuda.is_available())
            self.networks[img_key] = network

        self.shared_parameters.register(self)
        self.shared_parameters.proj()

    def save_model(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    def loss_for_img(self, img_key, input):
        network = self.networks[img_key]
        (val_dict, cond_mask_dict) = network.get_mapped_input_dict(np.array([ np.array([input]) ]))
        loss = network.ComputeTMMLoss(val_dict=val_dict, cond_mask_dict=cond_mask_dict)

        return loss

    def compute_total_loss(self, sample_img_key, per_network_loss):
        correct_nll= per_network_loss[sample_img_key]

        loss = 0
        margin = 1
        for img_key in self.img_keys:
            if img_key != sample_img_key:
                other_nll = per_network_loss[img_key]
                class_loss = (margin + correct_nll - other_nll).clamp(min=0)
                loss += class_loss

        return loss

    def train_discriminatively(self, num_sample):
        opt = optim.SGD( self.parameters() , lr=.0003)
        self.zero_grad()

        batch = 10
        total_loss = 0

        i = 0
        while i < num_sample:
            self.examples_trained += 1
            for sample_img_key in self.img_keys:
                input = segmented_data[sample_img_key][i]

                per_network_loss = {}
                for img_key in self.img_keys:
                    per_network_loss[img_key] = self.loss_for_img(img_key, input)

                loss = self.compute_total_loss(sample_img_key, per_network_loss)

                loss.backward()
                total_loss += loss

            if i % batch == 0 or i == num_sample - 1:
                print("Total loss: " + str(i) + " " + str(total_loss.data))

                if np.isnan(total_loss.data.cpu().numpy()):
                    return
                total_loss = 0
                opt.step()

                self.zero_grad()
                self.shared_parameters.proj()

            i += 1

    def train_on_img(self, img, img_key, num_iter):
        opt = optim.Adam( self.parameters() , lr=.05)
        self.zero_grad()

        batch = 1
        total_loss = 0

        i = 0
        while i < num_iter:
            self.examples_trained += 1
            loss = self.loss_for_img(img_key, img)

            loss.backward()
            total_loss += loss

            if i % batch == 0 or i == num_iter - 1:
                print("Total loss on sample " + str(i) + " :" + str(total_loss[0][0].data))

                if np.isnan(total_loss.data.cpu().numpy()):
                    return

                total_loss = 0
                opt.step()
                self.zero_grad()
                self.shared_parameters.proj()

            i += 1

    def train_generatively(self, num_sample):
        opt = optim.Adam( self.parameters() , lr=.003, weight_decay=0.001)
        self.zero_grad()

        batch = 10
        total_loss = 0

        i = 0
        while i < num_sample:
            self.examples_trained += 1
            for img_key in self.img_keys:
                input = segmented_data[img_key][i]

                loss = self.loss_for_img(img_key, input)

                loss.backward()
                total_loss += loss

            if i % batch == 0 or i == num_sample - 1:
                print("Total loss on sample " + str(i) + " :" + str(total_loss[0][0].data))

                if np.isnan(total_loss.data.cpu().numpy()):
                    return
                total_loss = 0
                opt.step()
                self.zero_grad()
                self.shared_parameters.proj()

            i += 1

def main():
    print("Loading dataset...")
    global segmented_data
    segmented_data = get_segmented_data()
    img_key = 2
    img = segmented_data[2][0]

    print("Creating SPN")
    tspn = TrainedConvSPN([img_key])

    print("Training SPN")
    tspn.train_on_img(img, img_key, 10)
    tspn.save_model('cifar_oneimg_' + str(img_key))

    pdb.set_trace()

if __name__ == '__main__':
    main()
