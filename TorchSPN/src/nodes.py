#!/usr/bin/env python3

import torch
from torch.autograd import Variable as Variable
import numpy as np
import pdb

EPSILON = 0.001

class Nodes(torch.nn.Module):
    '''
    Base class for SPN nodes (also called a layer).
    '''
    def __init__(self, is_cuda=False):
        '''
        Initialize nodes.
        :param is_cuda: True when computation should be done with CUDA (GPU).
        '''
        super(Nodes, self).__init__()

        self.is_cuda = is_cuda

    def var(self, tensor, requires_grad=False):
        '''
        Returns PyTorch Variable according to this node's settings.
        Currently only determines if the tensor is in GPU.
        :return: PyTorch Variable
        '''
        if self.is_cuda:
            tensor = tensor.cuda()

        return Variable(tensor, requires_grad)

class ConcatLayer(Nodes):
    def __init__(self, is_cuda, child_list):
        Nodes.__init__(self, is_cuda).__init__()
        self.child_list = child_list
        self.child_edges = []
        self.parent_edges = []
        self.num = 0
        for c in child_list:
            self.num += c.num

    def forward(self):
        self.val = torch.cat([c.val for c in self.child_list], dim=1)
        return self.val

class SumNodes(Nodes):
    '''
    The class of a set of sum nodes (also called a sum layer)
    '''
    def __init__(self, is_cuda, num=1):
        '''
        Initialize a set of sum nodes
        :param num: the number of sum nodes
        '''
        Nodes.__init__(self, is_cuda).__init__()
        self.num = num
        self.child_edges = []
        self.parent_edges = []
        self.scope = None  # todo
        self.is_cuda = is_cuda

    def gen_sample(self):
        child = self.child_edges[0]

        Y = self.samples
        # batch x Ny

        M = self.child_edges[0].mask
        W = self.child_edges[0].weights
        # Nx x Ny

        #print(Y)
        #print(torch.t(M * W ))

        # mask
        dist = torch.t(M * W )
        # batch x Nx
        n_batch = Y.size()[0]
        n_var   = self.child_edges[0].child.num


        x_onehot = []
        for idx, onedist in enumerate(dist):
            onedist = onedist.data.numpy()
            while sum(onedist) > 1:
                onedist *= .999
            sample_onedist = np.random.multinomial(1, onedist, n_batch)
            # batch x n_var

            x_onehot.append(sample_onedist)

        x_onehot = np.stack(x_onehot)
        # Ny x batch x Nx

        x_onehot = x_onehot.transpose([1,0,2])
        # batch x Ny x Nx

        samples = self.var( torch.FloatTensor(x_onehot.astype('float32')) )
        # batch x Ny x Nx

        samples = torch.bmm( Y.view(n_batch, 1, -1), samples)
        # batch x Ny x Nx

        samples = torch.sum(samples, dim =1)

        self.child_edges[0].child.samples = samples

    def forward(self):
        '''
        Overrides the method in torch.nn.Module
        :return: the value of this layer
        '''
        batch = self.child_edges[0].child.val.size()[0]

        self.val = self.var(torch.zeros(batch, self.num))
        # with size: <1 x Ny>

        # we are about to subtract the maximum value in all child nodes
        # to make sure the exp is operated on non-positive values
        for idx, e in enumerate(self.child_edges):
            tmpmax = torch.max( torch.max(e.child.val)) #  TODO: be careful when donig batched computation
            if idx == 0:
                maxval = tmpmax
            else:
                maxval = torch.max(maxval, tmpmax)
        maxval.detach() # disconnect during bp. any constant works here

        for e in self.child_edges:

            # e.child.val, size: <1 x Nx>
            # weights, size: <Nx x Ny>

            # log space computation:

            tmp = e.child.val - maxval # log(x/max)
            # with size <1 x Nx>

            tmp = torch.exp(tmp) # x/max
            # with size <1 x Nx>
            trueweights = e.weights * e.mask
            # with size <Nx x Ny>
            self.val += torch.mm(tmp, trueweights) # <w, x>/max)
            # with size <1 x Ny>


            # original space
            #self.val += torch.mm(e.child.val, e.weights)

        # log space only:

        self.val += torch.exp(torch.FloatTensor([-75]))[0]
        self.val = torch.log(self.val) # log( wx / max)
        self.val += maxval
        return self.val

class SparseProductNodes(Nodes):
    def __init__(self, is_cuda, num=1):
        Nodes.__init__(self, is_cuda).__init__()
        self.num = num
        self.child_edges = []
        self.parent_edges = []
        self.val = None

    def forward(self):
        batch = self.child_edges[0].child.val.size()[0]
        val = self.var(torch.zeros((batch, self.num)))

        # TODO: Implement batch forward

        for e in self.child_edges:
            child_val = e.child.val[0]

            for i in range(self.num):
                if e.connections[i]:
                    indices = e.connections[i]
                    selected_child = child_val[indices]

                    # if len(selected_child[selected_child == float('inf')]):
                        # pdb.set_trace()

                    val[0, i] = torch.sum(child_val[indices])

        self.val = val
        return val

class ProductNodes(Nodes):
    '''
    The class of a set of product nodes (also called a product layer)
    '''
    def __init__(self, is_cuda, num=1):
        '''
        Initialize a set of sum nodes
        :param num: the number of sum nodes
        '''
        Nodes.__init__(self, is_cuda).__init__()
        self.num = num
        self.child_edges = []
        self.parent_edges = []
        self.scope = None  # todo
        self.val = None
        self.samples = None

    def gen_sample(self):
        child = self.child_edges[0].child
        child.samples = torch.mm(self.samples, torch.t(self.child_edges[0].mask))

    def forward(self):
        '''
        Overrides the method in torch.nn.Module
        :return: the value of this layer
        '''
        # TODO: compute in the log space
        # however, we should first check the validity of log 0 for onehot leaf nodes
        batch = self.child_edges[0].child.val.size()[0]
        val = self.var(torch.zeros((batch, self.num)))
        # with size: batch x num_lower
        for e in self.child_edges:
            # log space
            val += torch.mm(e.child.val, e.mask)
            '''
            # original space
            num_child = e.child.num  # Is this variable going to be used?

            # child.val has size: <1 x Nx>
            # e.mask    has size: <Nx x Ny>  (suppose x -> y)
            tmp_val = torch.t(e.child.val.repeat(self.num, 1))
            # with size: <Nx x Ny>
            tmp_val = torch.pow(tmp_val, e.mask)
            # with size: <Nx x Ny>

            tmp_val = tmp_val.prod(0)
            # with size: <1 x Ny>
            val = val * tmp_val
            '''
        self.val = val
        return val


#######################
# Leaf nodes

class GaussianNodes(Nodes):
    '''
    The class of a set of Gaussian leaf nodes
    '''

    def __init__(self, is_cuda, mean, logstd):
        '''
        Initialize a set of Guassian nodes with parameters (mu, diag(sigma))
        :param mu:
        :param sigma:
        '''
        Nodes.__init__(self, is_cuda).__init__()

        self.num = 1          # the number of Gaussian nodes
        self.mean  = mean
        self.logstd = logstd
        self.is_cuda = is_cuda
        pass

    def gen_samples(self, num=2):
        self.samples = np.random.normal(float(self.mean.data.numpy()),
                         float(np.exp(self.logstd.data.numpy())),
                         (num,1)
                         )
        return self.samples
    def forward(self):
        '''
        Overrides the method in torch.nn.Module
        :return: the value of current layer
        compute log p(x; mu, sigma)
        '''
        self.input = self.var(torch.from_numpy(self.input.astype('float32')))
        x_mean = self.input - self.mean
        std    = torch.exp(self.logstd)
        var    = std * std

        self.val = (1 - self.marginalize_mask) * ( - (x_mean) * (x_mean) / 2.0 / var - self.logstd - 0.91893853320467267)
        # Note: if marginalized out, log p = 0

        return self.val
    def feed_val(self, x, marginalize_mask = None):
        self.input = x
        batch = x.shape[0]
        if marginalize_mask is None:
            # do not marginalize
            marginalize_mask = np.zeros((batch, self.num), dtype='float32')
        # else if marginalize_mask specified:
            # do nothing
        self.marginalize_mask = self.var(torch.from_numpy(marginalize_mask.astype('float32')))
        pass

    def feed_marginalize_mask(self, mask):
        self.marginalize_mask = self.var(torch.from_numpy(mask.astype('float32')))


    def mean_proj_hook(self):
        pass

    def std_proj_hook(self):
        if self.logstd.data.cpu().numpy().__float__() < -100:
            self.std.data = self.std.data.clamp(min=-85)


class MultinomialNodes(Nodes):
    '''
    The bernoulli nodes (binary variables with bernoulli distributions)
    '''
    def __init__(self, is_cuda, n_variable, n_out, n_values=2, list_n_values=None,
                 prob_matrix=None):
        '''
        Initialize a set of multinoulli nodes
        :param n_variable: the number of nodes
        :param n_out:      the number of probabilistic distribution imposed on each variable\
        :param n_values:   the number of values each variable takes (all variable take the same number of values)
        :param list_n_values: a list of numbers of values that a node takes
                Suppose we have r.v.s X1, X2, X3, taking 2, 5, 3 values resp.
                Then list_n_value is (2, 5, 3)
                `n_values` and `list_n_value` are mutual exclusive, i.e., can be specified for at most one
        :param prob_matrix: an instance of `torch.nn.Parameters`
                the probability matrices with size < n_value x (n_varialbe * n_out)>
        This class is implemented by `torch.nn.Embedding` for each variable.
        However, "embeddings" may not be shared across variables, so we explicit compute offset.
        '''
        # TODO: share embeddings

        Nodes.__init__(self, is_cuda).__init__()
        self.n_variable = n_variable
        self.n_out = n_out
        self.n_values = n_values
        self.list_n_values = list_n_values
        self.marginalize_mask = None
        self.num = n_out * n_variable  # dimension of output
        self.val = None     # output (n_variable)
        self.parent_edges = []

        # `self.offset` is the offset in the embedding matrix for table lookup
        if list_n_values is None:
            # If `n_values` is specified, all variables take a same number of values
            # E.g., for r.v.s X1, X2, X3 each taking 5 values,
            # Then n_variable = 3, n_values = 5, and the offset should be [0, 5, 10]
            # In this case, equivalent vocab_size = 15
            self.offset = np.arange(0, n_values * n_out, n_values, dtype='int32')
            self.vocab_size = n_values * n_out
            self.list_n_values = [n_values] * n_variable
        else:
            # If `list_n_values` is specified
            # Suppose r.v.s X1, X2, X3 taking 2, 5, 3 values, resp.
            # Then list_n_values is [2, 5, 3], and the offset should be [0, 2, 7]
            # In this case, equivalent vocab_size = 10
            self.offset = np.array([sum(list_n_values[:i]) for i in range(len(list_n_values))])
            self.vocab_size = sum(list_n_values)

        self.offset = self.offset.reshape((1, -1))

        self.prob_matrix = prob_matrix

        self.embed_layer = torch.nn.Embedding(self.vocab_size, n_out)
        self.embed_layer.weight = prob_matrix

    def forward(self):
        '''
        Overrides the method in torch.nn.Module
        :return: the value of current layer
        compute p(x; mu, sigma)
        '''


        # First pretend that none of the variables is masked to marginalize out

        # If embeddings are not shared over different r.v.s, we need to apply offset over input r.v.s
        input = self.input + self.offset
        input = self.var(torch.from_numpy(input.astype('int64')))
        # with size: <batch x n_variable>

        batch = input.size()[0]

        val_not_marginal = self.embed_layer(input).view(batch, -1)
        # embed_layer(*) yields a size of <1 x n_variable, n_embed_dim>
        # val has a size: <batch x (n_variable * n_embed_dim)>


        # Then pretend that all variables are to marginalize out

        m = self.marginalize_mask.view(batch, -1, 1)
        # with size: <batch x n_var x 1>

        m = m.repeat(1, 1, self.n_out)
        # with size: n_embed_dim x n_variable

        m = m.view((batch, -1))

        val_marginal = m.view((batch, -1))
        # with size: <batch x (n_variable * n_embed_dim)>

        # Example: If mask = [1 0; 0 1; 1 1] and n_embed_dim = 2
        # Then val_marginal = [1 1 0 0; 0 0 1 1; 1 1 1 1]


        # Apply mask to val_not_marginal

        # in log space
        self.val = torch.log(val_not_marginal * (1 - val_marginal) + val_marginal)
        # original space:
        #self.val = val_not_marginal * (1 - val_marginal) + val_marginal
        # with size: batch x (n_variable * n_embed_dim)
        return self.val

    def feed_val(self, x_id, marginalize_mask=None):
        '''
        Feed the value
        :param x_id: numpy array, <batch x num>
        :param marginalize_mask: np array with size <1 x n_variable>
        :return: None
        If mask[i] = 1, marginalize the i-th variable out
        If mask[i] = 0, do not marginalize it out
        :return:
        '''
        self.input = x_id
        batch = x_id.shape[0]

        if marginalize_mask is None:
            # do not marginalize
            marginalize_mask = np.zeros((batch, self.n_variable), dtype='float32')
        # else if marginalize_mask specified:
            # do nothing
        self.marginalize_mask = self.var(torch.from_numpy(marginalize_mask.astype('float32')))
        # with size: batch x n_variable

    def feed_marginalize_mask(self, mask):
        '''
        Feed the mask of marginalization
        '''
        self.marginalize_mask = self.var(torch.from_numpy(mask.astype('float32')))
        pass

    def para_proj_hook(self):
        #print('in hood')
        self.embed_layer.weight.data = self.embed_layer.weight.data.clamp(min=EPSILON)

        for off, nval in zip(self.offset[0], self.list_n_values):
            thispara = self.embed_layer.weight[off:off+nval]
            partition = torch.sum(thispara, dim=0, keepdim=True)
            self.embed_layer.weight.data[off:off+nval,:] = (thispara / partition).data
        pass


class BinaryNodes(Nodes):
    '''
    The class of a set of inary nodes
    '''
    def __init__(self, is_cuda, num):
        '''
        Initilize a set of binary nodes
        :param num: the number of binary nodes
        '''
        Nodes.__init__(self, is_cuda).__init__()
        self.num = num
        self.val = None
        self.parent_edges = []

    def feed_val(self, x_onehot=None, x_id=None):
        '''
        feed value to BinaryNodes
        :param x_onehot: one-hot representation of x, with size: <2 x num>
        :param x_id: ID representation of x, with size: num
        :return: None
        '''

        self.val = self.var(torch.from_numpy(x_onehot.astype('float32').reshape(1, -1)))
        self.val = torch.log(self.val)
        # wish size: <2 * num>
        # TODO: batch-ization: <batch x (2*num)>
        # TODO change ID representation to one hot

        pass

    def forward(self):
        '''
        Overrides the method in torch.nn.Module
        :return: the value of this layer
        '''
        return self.val


if __name__ == '__main__':

    pass
