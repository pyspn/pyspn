#!/usr/bin/env python3

import torch
import numpy as np
from torch.autograd import Variable as Var

EPSILON = 0.00001

class Param():
    '''
    Global status of the world
    Currently keeps tracks of parameters only
    '''
    para_list = None  # Parameter list

    def __init__(self):
        '''
        Initlize the parameter list
        '''
        self.para_list = torch.nn.ParameterList()
        self.hook_list = []
    def add_param(self, para, hook):
        '''
        Add a set of parameters to the world
        :param para: An instance of torch.nn.Parameters
        :param mask:
        :return: None
        If mask = None: do not normalize the corresponding parameter
        Else: normalize the parameter as a probabilistic distribution according to the
              variables indicated in the masks
        '''

        self.para_list.append(para)

        if hook is not None:
            self.hook_list.append(hook)

    def register(self, model):
        for idx, para in enumerate(self.para_list):
            
            model.register_parameter('para_' + str(idx), para)
            self.model = model

    def get_unrolled_para(self):
        pvector = np.array([])

        for para in self.para_list:
            pvector = np.concatenate( (pvector, para.data.numpy().reshape(-1)) )
        return pvector.reshape((-1,1))

    def get_unrolled_grad(self):
        pvector = np.array([])

        for para in self.para_list:
            pvector = np.concatenate( (pvector, para.grad.data.numpy().reshape(-1)) )

        return pvector.reshape( (-1,1) )

    def set_para(self, newpara):
        offset = 0

        for p in self.para_list:
            shape = p.size()
            num   = np.prod(shape)
            new_para = newpara[offset:offset+num].reshape(shape).astype('float32')
            offset += num

            p.data = torch.from_numpy(new_para)
        self.register(self.model)

    def proj(self):

        for idx, (p, hook) in enumerate(zip(self.para_list, self.hook_list)):
            hook()
        self.register(self.model)





if __name__ == '__main__':

    param = Param()
