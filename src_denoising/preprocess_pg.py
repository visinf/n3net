'''
Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

This file is part of the implementation as described in the NIPS 2018 paper:
Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
Please see the file LICENSE.txt for the license governing this code.
'''

import numpy as np
import random

import torch
from torch.autograd import Variable


def cfa_to_depth(T):
    Ts = []
    for yy in range(2):
        for xx in range(2):
            c = T[:,0,yy::2,xx::2]
            Ts.append(c)
    T_bayer = torch.stack(Ts, dim=1)
    return T_bayer

def depth_to_cfa(T):
    b,_,h,w = T.shape
    Tout = Variable(T.data.new(1,1,2*h,2*w), requires_grad=False)
    i = 0
    for yy in range(2):
        for xx in range(2):
            Tout[:,0,yy::2,xx::2] = T[:,i,:,:]
            i += 1
    return Tout

class Bayer(object):
    def __init__(self, factors):
        self.factors = factors

    def __call__(self, T):
        if np.random.rand() > 0.5:
            if np.random.rand() > 0.5:
                pattern = [0,1,1,2]
            else:
                pattern = [2,1,1,0]
        else:
            if np.random.rand() > 0.5:
                pattern = [1,0,2,1]
            else:
                pattern = [1,2,0,1]

        fac = np.random.rand(1)[0] * (self.factors[1] - self.factors[0]) + self.factors[0]
        Ts = []
        i=0
        for yy in range(2):
            for xx in range(2):
                c = T[pattern[i],yy::2,xx::2]
                if pattern[i] != 1:
                    fac_c = fac + random.normalvariate(0,0.01)
                    c = c * fac_c
                Ts.append(c)

        T_bayer = torch.stack(Ts, dim=0)

        return T_bayer

class PowerTransform(object):
    def __init__(self, gamma, gamma2=None):
        self.gamma = gamma
        self.gamma2 = gamma2

    def __call__(self, T):
        if self.gamma2 is not None:
            gamma = np.random.rand(1)[0] * (self.gamma2 - self.gamma) + self.gamma
        else:
            gamma = self.gamma
        return T**gamma

class DiscreteIntensityScale(object):
    def __init__(self, factors):
        self.factors = factors

    def __call__(self, T):
        return T*self.factors[np.random.randint(len(self.factors))]


class ContinuousIntensityScale(object):
    def __init__(self, factors):
        self.factors = factors

    def __call__(self, T):
        fac = np.random.rand(1)[0] * (self.factors[1] - self.factors[0]) + self.factors[0]
        return T*fac
