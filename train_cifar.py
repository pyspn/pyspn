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
        (val_dict, cond_mask_dict) = network.get_mapped_input_dict(np.array([input]))
        loss = network.ComputeTMMLoss(val_dict=val_dict, cond_mask_dict=cond_mask_dict)

        return loss

    def train_generatively(self, num_sample):
        opt = optim.Adam( self.parameters() , lr=.003)
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

                if np.isnan(total_loss[0][0].data.cpu().numpy()):
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
    img_keys = [0]

    print("Creating SPN")
    tspn = TrainedConvSPN(img_keys)
    print("Training SPN")
    tspn.train_generatively(100)
    tspn.save_model('cifar_' + str(img_keys))

    pdb.set_trace()

if __name__ == '__main__':
    main()
