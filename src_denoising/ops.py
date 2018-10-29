'''
Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

This file is part of the implementation as described in the NIPS 2018 paper:
Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
Please see the file LICENSE.txt for the license governing this code.
'''

from collections import namedtuple
import pyinn
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

try:
    import tensor_comprehensions as tc
except: pass

def has_tensor_comprehensions():
    return 'tc' in vars() and 'tc' in globals()

def indexed_matmul_1_tc(x,y,I, tune=False):
    if not has_tensor_comprehensions():
        return indexed_matmul_1(x, y ,I)

    lang = """
    def indexed_matmul_1_tc(float(B,N,E) X, float(B,M,E) Y, int32(B,M,O) I) -> (output) {
        output(b, m, o) +=! Y(b, m, j) * X(b, I(b,m,o), j)
    }   
    """

    b,m,_ = y.shape
    o = I.shape[2]
    n,e = x.shape[1:]
    cachefile  = "tc_kernels/b{}_m{}_o{}_e{}.tc".format(b,m,o,e)
    op = tc.define(lang, name="indexed_matmul_1_tc")
    if tune:
        tune_opt = tc.autotuner_settings
        tune_opt["cache"] = cachefile
        op.autotune(x,y,I.int(), **tune_opt)

    out = op(x,y,I.int(), cache=cachefile, options=tc.mapping_options.Options("naive"))
    if out is None:
        out = op(x,y,I.int(), options=tc.mapping_options.Options("naive"))
    return out


def indexed_matmul_1(x,y,I):
    b,m,_ = y.shape
    o = I.shape[2]
    e = x.shape[2]
    If = I.view(b, m*o,1).expand(b,m*o,e)
    x_ind = x.gather(dim=1, index=If)
    x_ind = x_ind.view(b,m,o,e)
    out = torch.matmul(x_ind, y.unsqueeze(3)).view(b,m,o)
    return out

def indexed_matmul_2_tc(x,y,I, tune=False):
    if not has_tensor_comprehensions():
        return indexed_matmul_2(x, y ,I)

    lang = """
    def indexed_matmul_2_tc(float(B,N,F) X, float(B,M,O,K) Y, int32(B,M,O) I) -> (output) {
        output(b, m, f, k) +=! Y(b, m, o, k) * X(b, I(b,m,o), f)
    }
    """
    b,m,_,k = y.shape
    o = I.shape[2]
    n,f = x.shape[1:]
    cachefile = "tc_kernels/b{}_m{}_o{}_k{}_f{}.tc".format(b,m,o,k,f)
    op = tc.define(lang, name="indexed_matmul_2_tc")
    if tune:
        tune_opt = tc.autotuner_settings
        tune_opt["cache"] = cachefile
        op.autotune(x,y,I.int(), **tune_opt)

    out = op(x,y,I.int(), cache=cachefile, options=tc.mapping_options.Options("naive"))
    if out is None:
        out = op(x,y,I.int(), options=tc.mapping_options.Options("naive"))
    return out

def indexed_matmul_2(x,y,I):
    b,m,o,k= y.shape
    e = x.shape[2]
    If = I.view(b, m*o, 1).expand(b,m*o,e)
    x_ind = x.gather(dim=1, index=If)
    x_ind = x_ind.view(b,m,o,e).permute(0,1,3,2)
    out = torch.matmul(x_ind, y)
    return out

def euclidean_distance(x,y):
    out = -2*torch.matmul(x, y)
    out += (x**2).sum(dim=-1, keepdim=True)
    out += (y**2).sum(dim=-2, keepdim=True)
    return out


def calc_padding(x, patchsize, stride, padding=None):
    if padding is None:
        xdim = x.shape[2:]
        padvert = -(xdim[0] - patchsize) % stride
        padhorz = -(xdim[1] - patchsize) % stride

        padtop = int(np.floor(padvert / 2.0))
        padbottom = int(np.ceil(padvert / 2.0))
        padleft = int(np.floor(padhorz / 2.0))
        padright = int(np.ceil(padhorz / 2.0))
    else:
        padtop = padbottom = padleft = padright = padding

    return padtop, padbottom, padleft, padright


def im2patch(x, patchsize, stride, padding=None, returnpadding=False):
    padtop, padbottom, padleft, padright = calc_padding(x, patchsize, stride, padding)
    xpad = F.pad(x, pad=(padleft, padright, padtop, padbottom))

    x2col = pyinn.im2col(xpad, [patchsize]*2, [stride]*2, [0,0])
    if returnpadding:
        return x2col, (padtop, padbottom, padleft, padright)
    else:
        return x2col

def patch2im(x_patch, patchsize, stride, padding):
    padtop, padbottom, padleft, padright = padding
    counts = pyinn.col2im(torch.ones_like(x_patch), [patchsize]*2, [stride]*2, [0,0])
    x = pyinn.col2im(x_patch.contiguous(), [patchsize]*2, [stride]*2, [0,0])

    x = x/counts

    x = x[:,:,padtop:x.shape[2]-padbottom, padleft:x.shape[3]-padright]
    return x

class Im2Patch(nn.Module):
    def __init__(self, patchsize, stride, padding=None):
        super(Im2Patch, self).__init__()
        self.patchsize = patchsize
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return im2patch(x, self.patchsize, self.stride, self.padding)

class Patch2Im(nn.Module):
    def __init__(self, patchsize, stride, padding=None):
        super(Patch2Im, self).__init__()
        self.patchsize = patchsize
        self.stride = stride
        self.padding = padding

    def forward(self, x_patch):
        return patch2im(x_patch, self.patchsize, self.stride, self.padding)

# This follows semantics of numpy.finfo.
_Finfo = namedtuple('_Finfo', ['eps', 'tiny'])
_FINFO = {
    torch.HalfStorage: _Finfo(eps=0.00097656, tiny=6.1035e-05),
    torch.FloatStorage: _Finfo(eps=1.19209e-07, tiny=1.17549e-38),
    torch.DoubleStorage: _Finfo(eps=2.22044604925e-16, tiny=2.22507385851e-308),
    torch.cuda.HalfStorage: _Finfo(eps=0.00097656, tiny=6.1035e-05),
    torch.cuda.FloatStorage: _Finfo(eps=1.19209e-07, tiny=1.17549e-38),
    torch.cuda.DoubleStorage: _Finfo(eps=2.22044604925e-16, tiny=2.22507385851e-308),
}


def _finfo(tensor):
    r"""
    Return floating point info about a `Tensor` or `Variable`:
    - `.eps` is the smallest number that can be added to 1 without being lost.
    - `.tiny` is the smallest positive number greater than zero
      (much smaller than `.eps`).
    Args:
        tensor (Tensor or Variable): tensor or variable of floating point data.
    Returns:
        _Finfo: a `namedtuple` with fields `.eps` and `.tiny`.
    """
    if isinstance(tensor, Variable):
        return _FINFO[tensor.data.storage_type()]
    else:
        return _FINFO[tensor.storage_type()]

def clamp_probs(probs):
    eps = _finfo(probs).eps
    return probs.clamp(min=eps, max=1 - eps)