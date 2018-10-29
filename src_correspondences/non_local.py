'''
Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

This file is part of the implementation as described in the NIPS 2018 paper:
Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
Please see the file LICENSE.txt for the license governing this code.
'''

import tensorflow as tf
import numpy as np

def nonlocal_block(x, nmatchplanes, nl_opt, embednet_opt, tempnet_opt=None, shared_embedding=True):
    with tf.variable_scope("embed"):
        xe = embednet(x, nout_planes=nmatchplanes, **embednet_opt)
        if shared_embedding:
            ye = embednet(x, nout_planes=nmatchplanes, **embednet_opt)
        else:
            ye = xe
    if tempnet_opt is not None:
        with tf.variable_scope("temp"):
            xt = embednet(x, nout_planes=1, **tempnet_opt)
    else:
        xt = None

    with tf.variable_scope("nonlocal"):
        xout = nonlocal_layer(x, xe, ye, xt=xt, **nl_opt)

    return xout

def nonlocal_layer(x, xe, ye, k=7, xt=None, is_training=False, exclude_self=False, distance_bn=True):
    print('nl_layer')

    print(xe.shape)

    D = -euclidean_distance(xe,tf.transpose(ye, perm=[0,1,3,2]))

    if distance_bn:
        D = tf.layers.batch_normalization(
                inputs=D,
                center=True, scale=True,
                training=is_training,
                trainable=True,
                axis=1
            )
    
    print(D.shape)
    Ws = meanfield_nn(D, k, xt, exclude_self=exclude_self)

    xs = [x]
    for W in Ws:
        sample = tf.matmul(W , x)
        print(W.shape)
        xs.append(sample)
        print(sample.shape)

    xout = tf.concat(xs, axis=-1)
    print(xout.shape)
    print('nl_layer end')
    return xout

def euclidean_distance(x,y):
    xsq = tf.reduce_sum(x**2, axis=-1, keepdims=True)
    ysq = tf.reduce_sum(y**2, axis=-2, keepdims=True)
    xty = -2*tf.matmul(x, y)

    out = xsq + ysq + xty
    return out

def meanfield_nn(D, k, temp=None, exclude_self=False):
    logits = D

    if temp is not None:
        logits = logits * temp # temp is actually treated as inverse temperature since this is numerically more stable
        print('with temp')


    if exclude_self:
        infs = tf.ones_like(logits[:,:,:,0]) * np.inf
        # infs = tf.ones_like(logits[:,:,:,0]) * (10000.0) # setting diagonal to -inf produces numerical problems ...
        logits = tf.matrix_set_diag(logits,-infs)

    W = []
    for i in range(k):
        weights_exp = tf.nn.softmax(logits, axis=-1)
        eps = 1.2e-7
        weights_exp = tf.clip_by_value(weights_exp, eps, 1-eps)
        W.append(weights_exp)
        logits = logits + tf.log1p(-weights_exp)

    return W

def embednet(x, conv_layer, nfeatures, nout_planes, depth):
    for i in range(depth-1):
        with tf.variable_scope("hidden_{}".format(i)):
            x = conv_layer(x, nfeatures)

    if depth > 0:
        with tf.variable_scope("hidden_{}".format(depth)):
            x = conv_layer(x, nout_planes)

    return x