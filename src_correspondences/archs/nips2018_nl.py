'''
Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

This file is part of the implementation as described in the NIPS 2018 paper:
Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
Please see the file LICENSE.txt for the license governing this code.
'''

import tensorflow as tf

from ops import conv1d_layer, conv1d_resnet_block
from non_local import nonlocal_block


def build_graph(x_in, is_training, config):

    activation_fn = tf.nn.relu

    x_in_shp = tf.shape(x_in)

    cur_input = x_in
    print(cur_input.shape)
    idx_layer = 0
    numlayer = config.net_depth
    ksize = 1
    nchannel = config.net_nchannel
    # Use resnet or simle net
    act_pos = config.net_act_pos
    conv1d_block = conv1d_resnet_block

    k = config.nl_k
    nmatchplanes = nchannel / (k+1)

    # First convolution
    with tf.variable_scope("hidden-input"):
        cur_input = conv1d_layer(
            inputs=cur_input,
            ksize=1,
            nchannel=nchannel,
            activation_fn=None,
            perform_bn=False,
            perform_gcn=False,
            is_training=is_training,
            act_pos="pre",
            data_format="NHWC",
        )
        print(cur_input.shape)
    for _ksize, _nchannel in zip(
            [ksize] * numlayer, [nchannel] * numlayer):
        scope_name = "hidden-" + str(idx_layer)
        idx_layer += 1

        with tf.variable_scope(scope_name):
            cur_input = conv1d_block(
                inputs=cur_input,
                ksize=_ksize,
                nchannel=nmatchplanes if idx_layer == numlayer // 2 else _nchannel,
                activation_fn=activation_fn,
                is_training=is_training,
                perform_bn=config.net_batchnorm,
                perform_gcn=config.net_gcnorm,
                act_pos=act_pos,
                data_format="NHWC",
            )
            # Apply pooling if needed
            print(cur_input.shape)

        if idx_layer == numlayer // 2:
            scope_name = scope_name + "_nl"
            with tf.variable_scope(scope_name):
                conv_layer = lambda x, nfeatures: conv1d_block(
                inputs=x,
                ksize=1,
                nchannel=nfeatures,
                activation_fn=activation_fn,
                perform_bn=config.nl_batchnorm,
                perform_gcn=config.nl_gcnorm,
                is_training=is_training,
                act_pos="pre",
                data_format="NHWC",
                )

                nl_opt = dict(k=k, exclude_self=config.nl_exclude_self, is_training=is_training, distance_bn=config.nl_distance_bn)
                embed_opt = dict(conv_layer=conv_layer, nfeatures=_nchannel, depth=3)
                if config.nl_tempnet:
                    tempnet_opt = dict(conv_layer=conv_layer, nfeatures=_nchannel, depth=3)
                else:
                    tempnet_opt = None
                cur_input = nonlocal_block(cur_input, nmatchplanes, nl_opt, embed_opt, tempnet_opt, shared_embedding=config.nl_shared_embedding)

    with tf.variable_scope("output"):
        cur_input = conv1d_layer(
            inputs=cur_input,
            ksize=1,
            nchannel=1,
            activation_fn=None,
            is_training=is_training,
            perform_bn=False,
            perform_gcn=False,
            data_format="NHWC",
        )
        #  Flatten
        cur_input = tf.reshape(cur_input, (x_in_shp[0], x_in_shp[2]))

    logits = cur_input
    print(cur_input.shape)

    return logits


#
# cvpr2018.py ends here
