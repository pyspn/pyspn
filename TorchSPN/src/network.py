#!/usr/bin/env python3

import torch
from torch.autograd import Variable as Variable

import os.path
import sys
import pdb

import time
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import nodes, edges, gl
import numpy as np

class Network(torch.nn.Module):
    '''
    A particular SPN structure, whose parameters are in Param
    '''

    def __init__(self, is_cuda = False):
        '''
        Initialize the lists of nodes and edges
        '''
        super(Network, self).__init__()
        self.leaflist = list()  # Leaf list or dict
        self.nodelist = list()  # Variable list
        self.edgelist = list()  # Edgelist
        self.is_cuda = is_cuda

        return

    def TopologicalSort(self):
        '''
        Topologically sort self.NodeList
        :return: None
        '''
        pass

    def forward(self):
        '''
        Override torch.nn.Module.forward
        :return: last layer's values
        '''
        # durations = []

        # leaf_start = time.time()
        for layer in self.leaflist:
            val = layer()
            if gl.debug:
                print('Leaf node: {}'.format(val.data.cpu().numpy()))
        # leaf_end = time.time()
        # leaf_duration = leaf_end - leaf_start

        # durations.append(leaf_duration)

        for layer in self.nodelist:
            # node_start = time.time()
            val = layer()
            # node_end = time.time()
            # node_duration = node_end - node_start
            # durations.append(node_duration)
            if gl.debug:
                print('Intermediate node: {}'.format(val.data.cpu().numpy()))

        # durations_by_type = defaultdict(float)
        #
        # for (i, duration) in enumerate(durations):
        #     if i == 0:
        #         node_type = "Leaf"
        #         durations_by_type[node_type] += duration
        #         print(str(i) + " " + node_type + " " + str(duration))
        #     else:
        #         node_type = str(type(self.nodelist[i - 1]))
        #         durations_by_type[node_type] += duration
        #         print(str(i) + " " + node_type + " " + str(duration))
        #
        # for node_type in durations_by_type:
        #     print(node_type + " :" + str(durations_by_type[node_type]))
        #
        # pdb.set_trace()

        return val

    def feed(self, val_dict={}, cond_mask_dict={}):
        '''
        Feed input values
        :param val_dict: A dictionary containing <variable, value> pairs
        :param cond_mask_dict: A dictionary containing <variable, mask> pairs
            A mask of 1 indicates the variable is conditioned
        :return: None
        For variables not in `cond_dict`, assume not conditioned (i.e., query variables)
        For variables not in `val_dict`, assume to be marginalized out (i.e., all ones)
        '''

        for k in val_dict:
            k.feed_val(val_dict[k])
        for k in cond_mask_dict:
            k.feed_marginalize_mask(cond_mask_dict[k])

    def AddConcatLayer(self, child_list):

        _nodes =  nodes.ConcatLayer(is_cuda=self.is_cuda, child_list=child_list)
        self.nodelist.append(_nodes)
        return _nodes

    def var(self, tensor, requires_grad=False):
        if self.is_cuda:
            tensor = tensor.cuda()
        return Variable(tensor, requires_grad=requires_grad)

    def parameter(self, tensor, requires_grad=False):
        if self.is_cuda:
            tensor = tensor.cuda()
        return torch.nn.Parameter(tensor, requires_grad=requires_grad)

    def AddMultinomialNodes(self,
                            n_variable,
                            n_out,
                            n_values=2,
                            list_n_values=None,
                            list_prob=None,
                            isReused=False,
                            parameters=None):
        '''
        Add a set of Bernoulli nodes
        :param num_variables: the number of variables
        :param num_value: the number of values each node takes
        :param list_prob: initial distribution of these nodes
        :param isReused: if the parameters are reused
        :param param: the global parameter set of SPN
        :return: the multinomial nodes
        '''

        if list_prob is None:
            print('not implemented')
        elif not isReused:
            Tall_lookup_table = np.concatenate(list_prob).astype('float32')
            # with size: < sum(all n_values for all r.vs)  x n_out>
            Tall_lookup_table = self.parameter(
                torch.from_numpy(Tall_lookup_table),
                requires_grad=True)
            parameters.add_param(Tall_lookup_table, None)
        # else if isReused
        # do nothing

        #        def __init__(self, n_variable, n_out, n_values=2, list_n_values=None, prob_matrix=None):

        _nodes = nodes.MultinomialNodes(
            is_cuda=self.is_cuda,
            n_variable=n_variable,
            n_out=n_out,
            n_values=n_values,
            list_n_values=list_n_values,
            prob_matrix=Tall_lookup_table)

        if not isReused:
            parameters.hook_list.append(_nodes.para_proj_hook)
        self.leaflist.append(_nodes)
        return _nodes

    def AddGaussianNodes(self, mean, std,
                            isReused=False,
                            parameters=None):
        '''
        Add a set of Bernoulli nodes
        :param mean: mean of Gaussian
        :param std:  std  of Gaussian
        :param isReused: if the parameters are reused
        :param param: the global parameter set of SPN
        :return: the Gaussian nodes
        For Node i, the probability of Node i being 1 is p_Bern[i].
        '''

        if isReused is None:
            print('not implemented')

        elif not isReused:
            mean = self.parameter(torch.from_numpy(mean), requires_grad=True)

            logstd  = self.parameter(torch.from_numpy(np.log(std)), requires_grad=True)
        # else if isReused
        # do nothing

        #        def __init__(self, n_variable, n_out, n_values=2, list_n_values=None, prob_matrix=None):

        _nodes = nodes.GaussianNodes(is_cuda=self.is_cuda, mean=mean, logstd=logstd)

        if not isReused:
            parameters.add_param(mean, _nodes.mean_proj_hook)
            parameters.add_param(logstd,  _nodes.std_proj_hook)
        self.leaflist.append(_nodes)
        return _nodes


    def AddBernoulliNodes(self, num, p_Bern=None, isReused=False, param=None):
        '''
        Add a set of Bernoulli nodes
        :param num: the number of nodes
        :param p_Bern: instance of `torch.nn.Parameters`, initial distribution of these nodes
        :param isReused: if the parameters are reused
        :param param: the global status of param
        :return: the constructed Bernoulli nodes

        For Node i, the probability of Node i being 1 is p_Bern[i].
        If `isReused`, `p_Bern` is an instance of `nn.torch.Parameters` and has already been tracked in `param`
        If not `isReused`, `p_Bern` is a numpy array; we create `nn.torch.Parameters` and add it to `param`
        '''
        if p_Bern is None:
            # TODO: initialize according to some criterion
            p_Bern = np.ones((2))

    def AddBinaryNodes(self, num):
        '''
        Add a set of binary nodes to Network
        :param num: the number of nodes
        :return: An instance of Nodes.BinaryNodes
        '''
        _nodes = nodes.BinaryNodes(is_cuda=self.is_cuda, num=num)
        self.leaflist.append(_nodes)
        return _nodes

    def AddSumNodes(self, num):
        '''
        Add a set of sum nodes to Network
        :param num: the number of sum nodes
        :return: An instance of Nodes.SumNodes
        '''
        _nodes = nodes.SumNodes(is_cuda=self.is_cuda, num=num)
        self.nodelist.append(_nodes)
        return _nodes

    def AddSparseSumNodes(self, num):
        _nodes = nodes.SparseSumNodes(is_cuda=self.is_cuda, num=num)
        self.nodelist.append(_nodes)
        return _nodes

    def AddSparseProductNodes(self, num):
        _nodes = nodes.SparseProductNodes(is_cuda=self.is_cuda, num=num)
        self.nodelist.append(_nodes)
        return _nodes

    def AddProductNodes(self, num):
        '''
        Add a set of product nodes to Network
        :param num: the number of product nodes
        :return: An instance of Nodes.ProductNodes
        '''
        _nodes = nodes.ProductNodes(is_cuda=self.is_cuda, num=num)
        self.nodelist.append(_nodes)
        return _nodes

    def AddSparseSumEdges(
            self,
            lower,
            upper,
            mask_matrix,
            weight_matrix,
            parameters=None):
        _edges = edges.SparseSumEdges(lower, upper, mask_matrix, weight_matrix)
        upper.child_edges.append(_edges)
        lower.parent_edges.append(_edges)

        flattened_indices = self.var(_edges.flattened_indices)
        weights = self.parameter(_edges.connection_weights, requires_grad=True)

        parameters.add_param(weights, _edges.sum_weight_hook)

        return _edges

    def AddSparseProductEdges(self, lower, upper, mask_matrix):
        _edges = edges.SparseProductEdges(lower, upper, mask_matrix)
        upper.child_edges.append(_edges)
        lower.parent_edges.append(_edges)

        return _edges

    def AddProductEdges(self, lower, upper, mask=None):
        '''
        Add an edge: lower -> upper with weights weight
        :param lower: Instance of Nodes, lower layer
        :param upper: INstance of Nodes, upper layer
        :param mask: numpy aray with size <N_lower x N_upper>,
         masking out non-connected edges (a mask of 0 indicates disconnected edge)
        :return: An instance of Edges.ProductEdges
        '''
        if mask is None:
            mask = self.var(torch.ones((lower.num, upper.num)))
        else:
            mask = self.var(torch.from_numpy(mask.astype('float32')))
        _edges = edges.ProductEdges(lower, upper, mask)
        upper.child_edges.append(_edges)
        lower.parent_edges.append(_edges)

        return _edges

    def AddSumEdges(self,
                    lower,
                    upper,
                    weights=None,
                    mask=None,
                    isReused=False,
                    parameters=None):
        '''
        Add an edge: lower -> upper with weights weight
        :param lower: lower layer
        :param upper: upper layer
        :param weights: size N_lower x N_upper.
             `weights` is a numpy array if `isReused` is `False`
             `weights` is a instance of `torch.nn.Parameters` if `isReused` in `True`
             if `weights` is `None`, randomly inistalize weights according to some criterion
        :param mask: numpy aray with size <N_lower x N_upper> masking out non-connected edges (a mask of 0 indicates disconnected edge)
        :param isReused: Boolean. If is reused, the parameters are not added to `param`
        :param param: An instance of Param
        :return: (edge, para)
         edge is an instance of Edges.SumEdges
         para is an instance of torch.nn.Parameters
        '''

        if weights is None:
            # TODO: initialize weights according to some criterion
            pass
        elif not isReused:
            if mask is not None:
                mask = self.var(torch.from_numpy(mask), requires_grad = False)
            else: #  if mask is None:
                mask = self.var(torch.from_numpy(np.ones(weights.shape).astype('float32'))).detach()

            weights = self.parameter(torch.from_numpy(weights), requires_grad=True)

        # else isReused:
        # weights are already an instance of `torch.nnParameters`, and have been in `param.para_list`
        # do nothing


        _edges = edges.SumEdges(lower, upper, weights, mask)
        upper.child_edges.append(_edges)
        lower.parent_edges.append(_edges)

        if not isReused:
            parameters.add_param(weights, _edges.sum_weight_hook)


        return _edges, weights

    def ComputeUnnormalized(self, val_dict=None, marginalize_dict={}):
        '''
        Compute unnormalized measure
        :param val_dict: A dictionary containing <variable, value> pairs
            X U Y = variables
        :param cond_mask_dict: A dictionary containing <variable, mask> pairs
            A mask of 1 indicates the variable is conditioned (i.e., X)
            A mask of 0 indicates the variable is unconditioned (i.e., Y)
        :return: a scalar, the unnormalized measure p_tilde(Y|X)

        For variables not in `cond_mask_dict`, assume not conditioned (i.e., Y)
        For variables not in `val_dict`, assumed to be marginalized out
        '''
        self.feed(val_dict, marginalize_dict)
        return np.exp(self().data.cpu().numpy())

    def ComputeLogUnnormalized(self, val_dict=None, marginalize_dict={}):
        '''
        Compute unnormalized measure
        :param val_dict: A dictionary containing <variable, value> pairs
            X U Y = variables
        :param cond_mask_dict: A dictionary containing <variable, mask> pairs
            A mask of 1 indicates the variable is conditioned (i.e., X)
            A mask of 0 indicates the variable is unconditioned (i.e., Y)
        :return: a scalar, the unnormalized measure p_tilde(Y|X)

        For variables not in `cond_mask_dict`, assume not conditioned (i.e., Y)
        For variables not in `val_dict`, assumed to be marginalized out
        '''

        self.feed(val_dict, marginalize_dict)

        return self()

    def ComputeProbability(self, val_dict=None, cond_mask_dict={}, grad=False, log=False, is_negative=False):
        '''
        Compute unnormalized measure
        :param val_dict: A dictionary containing <variable, value> pairs
            X U Y = variables
        :param cond_mask_dict: A dictionary containing <variable, mask> pairs
            A mask of 1 indicates the variable is conditioned (i.e., X)
            A mask of 0 indicates the variable is unconditioned (i.e., Y)
        :return: a scalar, the unnormalized measure p_tilde(Y|X)

        For variables not in `cond_mask_dict`, assume not conditioned (i.e., Y)
        For variables not in `val_dict`, assumed to be marginalized out
        '''
        # TODO Not implemented: cond_mask_dict and/or val_dict not complete

        if gl.debug:
            print('-------- p_tilde ----------')
        log_p_tilde = self.ComputeLogUnnormalized(val_dict)

        marginalize_dict = {}
        for k in cond_mask_dict:
            marginalize_dict[k] = 1 - cond_mask_dict[k]
        if gl.debug:
            print('-------- Z ----------')

        log_Z = self.ComputeLogUnnormalized(val_dict, marginalize_dict)

        #print( log_p_tilde)
        #print(log_Z)

        J = torch.sum(- log_p_tilde + log_Z) #  negative log-likelihood

        #print('log p_tilder', log_p_tilde)
        #print('log z',        log_Z)
        #print('diff',         log_p_tilde - log_Z)
        #print('np.exp( log_p_tilde.data.numpy() - log_Z.data.numpy() )', np.exp( log_p_tilde.data.numpy() - log_Z.data.numpy() ))

        if grad:
            J.backward()

        prob = log_p_tilde.data.cpu().numpy() - log_Z.data.cpu().numpy()
        if not log:
            prob = np.exp(prob)
        return prob


    def CleanAllValues(self):
        '''
        Clean the activation functions of Nodes.val
        :return: None
        This function should be called before feeding a new data instance to a reused structure
        '''

        for node in self.leaflist:
            node.val = None
        for node in self.nodelist:
            node.val = None
        pass

    def UpdateWeight(self):
        '''
        A little bit vague for the current time being
        :return:
        '''
        pass

    def UpdateStruct(self):
        '''
        Too vague for the current time being
        :return:
        '''
        pass

    def AddRecurrentTemplate(self, input_list):
        '''
        Somehow vague for the current time being
        A template network with a list of input nodes
        This method updates network's internal information, including
        LeafList, NodeList, EdgeList, torchNet
        :param input_list:
        :return: a list of output nodes
        '''
        pass

    def GenSample(self, node2var, n_batch=2, n_var=4):
        # TODO: does not work with general framework for the time being
        self.nodelist[-1].samples= self.var(torch.ones((n_batch, 1)))
        for i in range(len(self.nodelist)-1, 0, -1):
            self.nodelist[i].gen_sample()
            print(self.nodelist[i].samples)
        print(self.nodelist[0].samples)
        leaf_indicator = self.nodelist[0].samples.data.cpu().numpy()
        leaf_value     = [ leaf.gen_samples(n_batch) for leaf in self.leaflist]
        leaf_value = np.concatenate(leaf_value,axis=1)

        where_dim0, where_dim1_node = np.where(leaf_indicator > 0)
        where_dim1_var = [ node2var[idx] for idx in where_dim1_node]

        samples = np.zeros((n_batch, n_var))
        samples[ where_dim0, where_dim1_var] = leaf_value[where_dim0, where_dim1_node]


        return samples
