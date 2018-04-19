#!/usr/bin/env python3

import torch
import numpy as np

EPSILON = 0.00001


class ProductEdges():
    '''
    # the class for a set of edges
    '''
    def __init__(self, child, parent, mask):
        '''
        Initialize a set of product edges
        :param child: child layer, with size: num_in
        :param parent: parent layer, with size: num_out
        :param mask: masks out unconnected edges, with size: num_in x num_out
        '''
        self.parent = parent
        self.child = child
        self.mask = mask


class SumEdges():
    '''
    # the class for a set of edges
    '''
    def __init__(self, child, parent, weights, mask):
        '''
        Initialize a set of product edges
        :param child: child layer, with size: num_in
        :param parent: parent layer, with size: num_out
        :param weights: size: num_in x num_out
        :param mask: masks out unconnected edges, with size: num_in x num_out
        '''
        self.parent = parent
        self.child = child
        self.weights = weights
        self.mask  = mask

    def sum_weight_hook(self):
        self.weights.data = self.weights.data.clamp(min=EPSILON)

        partition = torch.sum(self.weights.data * self.mask.data, dim=0, keepdim=True)
        self.weights.data /= partition
