#! /usr/bin/python
# -*- coding: utf8 -*-

###################################################################################################
#### This code was developed by Mariana Afonso, Phd student @ University of Bristol, UK, 2018 #####
################################## All rights reserved © ##########################################

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import tensorflow.contrib.slim as slim
from tensorlayer.layers_ESRGAN import *

def Dense_Block(layer1, layer2, layer3, C_nums=4, reuse=False):
    """ Generator for image SR
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    inputs = layer1
    i = layer2
    j = layer3
    with tf.variable_scope("Dense_Block", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        for k in range(C_nums):
            x = Conv2d(inputs, 16, (3, 3), (1, 1), padding='SAME', act=tf.nn.leaky_relu, name='dense_conv_RDBoutputs_{}_{}_{}'.format(i, j, k))
            inputs = ConcatLayer([x, inputs], concat_dim=3, name='dense_conv_RDBinputs_{}_{}_{}'.format(i, j, k))

        temp = Conv2d(inputs, 32, (1, 1), (1, 1), padding='SAME', act=tf.nn.leaky_relu, name='dense_conv_RDBoutputs_Rethinking_{}_{}_{}'.format(i, j, k))
        inputs = Conv2d(inputs, 32, (3, 3), (1, 1), padding='SAME', act=tf.identity, name='dense_conv_RDBfinal_{}_{}'.format(i, j))
        inputs.outputs = tf.multiply(inputs.outputs, 0.2)  # residual scaling
        inputs = ElementwiseLayer([layer1, inputs], tf.add, name='dense_final_{}_{}'.format(i, j))

        return inputs, temp

def Dense_Block_ResNet(layer1, layer2, layer3, C_nums=4, reuse=False):
    """ Generator for image SR
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    inputs = layer1
    i = layer2
    j = layer3
    with tf.variable_scope("Dense_Block_ResNet", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        for k in range(C_nums):
            x = Conv2d(inputs, 16, (3, 3), (1, 1), padding='SAME', act=tf.nn.leaky_relu, name='dense_conv_RDBoutputs_{}_{}_{}'.format(i, j, k))
            inputs = ConcatLayer([x, inputs], concat_dim=3, name='dense_conv_RDBinputs_{}_{}_{}'.format(i, j, k))

        #temp = Conv2d(inputs, 32, (1, 1), (1, 1), padding='SAME', act=tf.nn.leaky_relu, name='dense_conv_RDBoutputs_Rethinking_{}_{}_{}'.format(i, j, k))
        n = inputs
        channel = n.outputs.get_shape()[-1]
        for t in range(2):
            temp = Conv2d(n, channel, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', name='temp_conv1_{}_{}_{}'.format(i, j, t))
            temp = Conv2d(temp, channel, (3, 3), (1, 1), act=None, padding='SAME', name='temp_conv2_{}_{}_{}'.format(i, j, t))
            temp = ElementwiseLayer([n, temp], tf.add, name='temp_conv3_{}_{}_{}'.format(i, j, t))
            n = temp

        temp = Conv2d(temp, 64, (1, 1), (1, 1), padding='SAME', act=tf.identity, name='temp_final_{}_{}'.format(i, j))


        inputs = Conv2d(inputs, 64, (3, 3), (1, 1), padding='SAME', act=tf.identity, name='dense_conv_RDBfinal_{}_{}'.format(i, j))
        inputs.outputs = tf.multiply(inputs.outputs, 0.2)  # residual scaling
        inputs = ElementwiseLayer([layer1, inputs], tf.add, name='dense_final_{}_{}'.format(i, j))

        return inputs, temp

def Dense_Block_SFT(layer1, layer2, layer3, C_nums=4, reuse=False):
    """ Generator for image SR
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    inputs = layer1
    i = layer2
    j = layer3
    with tf.variable_scope("Dense_Block_SFT", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        for k in range(C_nums):
            x = Conv2d(inputs, 16, (3, 3), (1, 1), padding='SAME', act=tf.nn.leaky_relu, name='dense_conv_RDBoutputs_{}_{}_{}'.format(i, j, k))
            inputs = ConcatLayer([x, inputs], concat_dim=3, name='dense_conv_RDBinputs_{}_{}_{}'.format(i, j, k))

        temp = SFT(inputs, i, j)
        inputs = Conv2d(inputs, 32, (3, 3), (1, 1), padding='SAME', act=tf.identity, name='dense_conv_RDBfinal_{}_{}'.format(i, j))
        inputs.outputs = tf.multiply(inputs.outputs, 0.2)  # residual scaling
        inputs = ElementwiseLayer([layer1, inputs], tf.add, name='dense_final_{}_{}'.format(i, j))

        return inputs, temp

def Dense_Block_SFT_ResNet(layer1, layer2, layer3, C_nums=4, reuse=False):
    """ Generator for image SR
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    inputs = layer1
    i = layer2
    j = layer3
    with tf.variable_scope("Dense_Block_SFT_ResNet", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        for k in range(C_nums):
            x = Conv2d(inputs, 16, (3, 3), (1, 1), padding='SAME', act=tf.nn.leaky_relu, name='dense_conv_RDBoutputs_{}_{}_{}'.format(i, j, k))
            inputs = ConcatLayer([x, inputs], concat_dim=3, name='dense_conv_RDBinputs_{}_{}_{}'.format(i, j, k))

        temp = SFT_ResNet(inputs, i, j)
        inputs = Conv2d(inputs, 32, (3, 3), (1, 1), padding='SAME', act=tf.identity, name='dense_conv_RDBfinal_{}_{}'.format(i, j))
        inputs.outputs = tf.multiply(inputs.outputs, 0.2)  # residual scaling
        inputs = ElementwiseLayer([layer1, inputs], tf.add, name='dense_final_{}_{}'.format(i, j))

        return inputs, temp

def Dense_Block_Rethinking(layer1, layer2, layer3, layer4, C_nums=4, reuse=False):
    """ Generator for image SR
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    inputs1 = layer1
    inputs2 = layer2
    i = layer3
    j = layer4
    with tf.variable_scope("Dense_Block_Rethinking", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        for k in range(C_nums):
            x = Conv2d(inputs1, 16, (3, 3), (1, 1), padding='SAME', act=tf.nn.leaky_relu, name='dense_conv_RDBoutputs2_{}_{}_{}'.format(i, j, k))
            inputs1 = ConcatLayer([x, inputs1], concat_dim=3, name='dense_conv_RDBinputs2_{}_{}_{}'.format(i, j, k))

        inputs1 = ConcatLayer([inputs1, inputs2], concat_dim=3, name='dense_conv_RDBinputs_ConcatRethinking2_{}_{}_{}'.format(i, j, k))
        temp2 = Conv2d(inputs1, 32, (1, 1), (1, 1), padding='SAME', act=tf.nn.leaky_relu, name='dense_conv_RDBoutputs_Rethinking2_{}_{}_{}'.format(i, j, k))
        inputs1 = Conv2d(inputs1, 32, (3, 3), (1, 1), padding='SAME', act=tf.identity, name='dense_conv_RDBfinal2_{}_{}'.format(i, j))
        inputs1.outputs = tf.multiply(inputs1.outputs, 0.2)  # residual scaling
        inputs1 = ElementwiseLayer([layer1, inputs1], tf.add, name='dense_final2_{}_{}'.format(i, j))

        return inputs1, temp2

def Dense_Block_Rethinking_ResNet(layer1, layer2, layer3, layer4, C_nums=4, reuse=False):
    """ Generator for image SR
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    inputs1 = layer1
    inputs2 = layer2
    i = layer3
    j = layer4
    with tf.variable_scope("Dense_Block_Rethinking_ResNet", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        for k in range(C_nums):
            x = Conv2d(inputs1, 16, (3, 3), (1, 1), padding='SAME', act=tf.nn.leaky_relu, name='dense_conv_RDBoutputs2_{}_{}_{}'.format(i, j, k))
            inputs1 = ConcatLayer([x, inputs1], concat_dim=3, name='dense_conv_RDBinputs2_{}_{}_{}'.format(i, j, k))

        inputs1 = ConcatLayer([inputs1, inputs2], concat_dim=3, name='dense_conv_RDBinputs_ConcatRethinking2_{}_{}_{}'.format(i, j, k))
        m = inputs1
        channel = m.outputs.get_shape()[-1]
        #temp2 = Conv2d(inputs1, 32, (1, 1), (1, 1), padding='SAME', act=tf.nn.leaky_relu, name='dense_conv_RDBoutputs_Rethinking2_{}_{}_{}'.format(i, j, k))
        for t in range(2):
            temp2 = Conv2d(m, channel, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', name='temp2_conv1_{}_{}_{}'.format(i, j, t))
            temp2 = Conv2d(temp2, channel, (3, 3), (1, 1), act=None, padding='SAME', name='temp2_conv2_{}_{}_{}'.format(i, j, t))
            temp2 = ElementwiseLayer([m, temp2], tf.add, name='temp2_conv3_{}_{}_{}'.format(i, j, t))
            m = temp2

        temp2 = Conv2d(temp2, 64, (1, 1), (1, 1), padding='SAME', act=tf.identity, name='temp2_final_{}_{}'.format(i, j))

        inputs1 = Conv2d(inputs1, 64, (3, 3), (1, 1), padding='SAME', act=tf.identity, name='dense_conv_RDBfinal2_{}_{}'.format(i, j))
        inputs1.outputs = tf.multiply(inputs1.outputs, 0.2)  # residual scaling
        inputs1 = ElementwiseLayer([layer1, inputs1], tf.add, name='dense_final2_{}_{}'.format(i, j))

        return inputs1, temp2

def Dense_Block_Rethinking_SFT(layer1, layer2, layer3, layer4, C_nums=4, reuse=False):
    """ Generator for image SR
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    inputs1 = layer1
    inputs2 = layer2
    i = layer3
    j = layer4
    with tf.variable_scope("Dense_Block_Rethinking_SFT", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        for k in range(C_nums):
            x = Conv2d(inputs1, 16, (3, 3), (1, 1), padding='SAME', act=tf.nn.leaky_relu, name='dense_conv_RDBoutputs2_{}_{}_{}'.format(i, j, k))
            inputs1 = ConcatLayer([x, inputs1], concat_dim=3, name='dense_conv_RDBinputs2_{}_{}_{}'.format(i, j, k))

        inputs1 = ConcatLayer([inputs1, inputs2], concat_dim=3, name='dense_conv_RDBinputs_ConcatRethinking2_{}_{}_{}'.format(i, j, k))
        temp2 = SFT(inputs1, i, j)
        inputs1 = Conv2d(inputs1, 32, (3, 3), (1, 1), padding='SAME', act=tf.identity, name='dense_conv_RDBfinal2_{}_{}'.format(i, j))
        inputs1.outputs = tf.multiply(inputs1.outputs, 0.2)  # residual scaling
        inputs1 = ElementwiseLayer([layer1, inputs1], tf.add, name='dense_final2_{}_{}'.format(i, j))

        return inputs1, temp2

def Dense_Block_Rethinking_SFT_ResNet(layer1, layer2, layer3, layer4, C_nums=4, reuse=False):
    """ Generator for image SR
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    inputs1 = layer1
    inputs2 = layer2
    i = layer3
    j = layer4
    with tf.variable_scope("Dense_Block_Rethinking_SFT_ResNet", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        for k in range(C_nums):
            x = Conv2d(inputs1, 16, (3, 3), (1, 1), padding='SAME', act=tf.nn.leaky_relu, name='dense_conv_RDBoutputs2_{}_{}_{}'.format(i, j, k))
            inputs1 = ConcatLayer([x, inputs1], concat_dim=3, name='dense_conv_RDBinputs2_{}_{}_{}'.format(i, j, k))

        inputs1 = ConcatLayer([inputs1, inputs2], concat_dim=3, name='dense_conv_RDBinputs_ConcatRethinking2_{}_{}_{}'.format(i, j, k))
        temp2 = SFT_ResNet(inputs1, i, j)
        inputs1 = Conv2d(inputs1, 32, (3, 3), (1, 1), padding='SAME', act=tf.identity, name='dense_conv_RDBfinal2_{}_{}'.format(i, j))
        inputs1.outputs = tf.multiply(inputs1.outputs, 0.2)  # residual scaling
        inputs1 = ElementwiseLayer([layer1, inputs1], tf.add, name='dense_final2_{}_{}'.format(i, j))

        return inputs1, temp2

def RRDB_ShareSkip_1(layer1, layer2, C_nums=4, reuse=False):
    """ Generator for image SR
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    inputs1 = layer1
    i = layer2
    w_init = tf.variance_scaling_initializer(scale=0.01, seed=7)
    channel = inputs1.outputs.get_shape()[-1]
    with tf.variable_scope("RRDB_ShareSkip_1", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        for j in range(3):
            if j == 0:
                x = Dense_Block_ResNet(inputs1, i, j)  #
            else:
                x = Dense_Block_Rethinking_ResNet(x[0], x[1], i, j)  #
            if j <= 1:
                x = list(x)
                x[0] = ElementwiseLayer([x[0], inputs1], tf.add, name='RRDB_share_skip1_add_{}_{}'.format(i, j))  # Share-source Skip Connections
                x = tuple(x)

        x[0].outputs = tf.multiply(x[0].outputs, 0.2) #residual scale

        x = list(x)
        x[0] = ElementwiseLayer([x[0], inputs1], tf.add, name='RCB_conv1_final_{}_{}'.format(i, 3))
        x = tuple(x)

        return x[0], x[1]

def RRDB_ShareSkip_1_SFTResNet(layer1, layer2, C_nums=4, reuse=False):
    """ Generator for image SR
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    inputs1 = layer1
    i = layer2
    w_init = tf.variance_scaling_initializer(scale=0.01, seed=7)
    channel = inputs1.outputs.get_shape()[-1]
    with tf.variable_scope("RRDB_ShareSkip_1_SFTResNet", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        for j in range(3):
            if j == 0:
                x = Dense_Block_SFT_ResNet(inputs1, i, j)  #
            else:
                x = Dense_Block_Rethinking_SFT_ResNet(x[0], x[1], i, j)  #
            if j <= 1:
                x = list(x)
                x[0] = ElementwiseLayer([x[0], inputs1], tf.add, name='RRDB_share_skip1_SFTResNet_add_{}_{}'.format(i, j))  # Share-source Skip Connections
                x = tuple(x)

        x[0].outputs = tf.multiply(x[0].outputs, 0.2) #residual scale

        x = list(x)
        x[0] = ElementwiseLayer([x[0], inputs1], tf.add, name='RCB_conv1_SFTResNet_final_{}_{}'.format(i, 3))
        x = tuple(x)

        return x[0], x[1]

def RRDB_Cascading_1(layer1, layer2, C_nums=4, reuse=False):
    """ Generator for image SR
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    inputs1 = layer1
    i = layer2
    w_init = tf.variance_scaling_initializer(scale=0.01, seed=7)
    channel = inputs1.outputs.get_shape()[-1]
    with tf.variable_scope("RRDB_Cascading_1", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        RRDB_layer_1 = Dense_Block(inputs1, i, 1)
        RRDB_layer_1_c1 = ConcatLayer([RRDB_layer_1[0], inputs1], concat_dim=3, name='RRDB_layer_1_c1_{}_{}'.format(i, 1))  #
        RRDB_layer_1_c1 = Conv2d(RRDB_layer_1_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='RCB_conv1_{}_{}'.format(i, 1))  #

        RRDB_layer_2 = Dense_Block_Rethinking(RRDB_layer_1_c1, RRDB_layer_1[1], i, 2)  #
        RRDB_layer_2_c1 = ConcatLayer([RRDB_layer_2[0], RRDB_layer_1[0], inputs1], concat_dim=3, name='RRDB_layer_1_c1_{}_{}'.format(i, 2))  #
        RRDB_layer_2_c1 = Conv2d(RRDB_layer_2_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='RCB_conv1_{}_{}'.format(i, 2))  #

        RRDB_layer_3 = Dense_Block_Rethinking(RRDB_layer_2_c1, RRDB_layer_2[1], i, 3)  #
        RRDB_layer_3_c1 = ConcatLayer([RRDB_layer_3[0], RRDB_layer_2[0], RRDB_layer_1[0], inputs1], concat_dim=3, name='RRDB_layer_1_c1_{}_{}'.format(i, 3))  #
        RRDB_layer_3_c1 = Conv2d(RRDB_layer_3_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='RCB_conv1_{}_{}'.format(i, 3))  #

        RRDB_layer_3_c1 = ElementwiseLayer([RRDB_layer_3_c1, inputs1], tf.add, name='RCB_conv1_final_{}_{}'.format(i, 3))

        return RRDB_layer_3_c1, RRDB_layer_3[1]

def RRDB_Cascading_1_SFT(layer1, layer2, C_nums=4, reuse=False):
    """ Generator for image SR
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    inputs1 = layer1
    i = layer2
    w_init = tf.variance_scaling_initializer(scale=0.01, seed=7)
    channel = inputs1.outputs.get_shape()[-1]
    with tf.variable_scope("RRDB_Cascading_1_SFT", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        RRDB_layer_1 = Dense_Block_SFT(inputs1, i, 1)
        RRDB_layer_1_c1 = ConcatLayer([RRDB_layer_1[0], inputs1], concat_dim=3, name='RRDB_layer_1_c1_{}_{}'.format(i, 1))  #
        RRDB_layer_1_c1 = Conv2d(RRDB_layer_1_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='RCB_conv1_{}_{}'.format(i, 1))  #

        RRDB_layer_2 = Dense_Block_Rethinking_SFT(RRDB_layer_1_c1, RRDB_layer_1[1], i, 2)  #
        RRDB_layer_2_c1 = ConcatLayer([RRDB_layer_2[0], RRDB_layer_1[0], inputs1], concat_dim=3, name='RRDB_layer_1_c1_{}_{}'.format(i, 2))  #
        RRDB_layer_2_c1 = Conv2d(RRDB_layer_2_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='RCB_conv1_{}_{}'.format(i, 2))  #

        RRDB_layer_3 = Dense_Block_Rethinking_SFT(RRDB_layer_2_c1, RRDB_layer_2[1], i, 3)  #
        RRDB_layer_3_c1 = ConcatLayer([RRDB_layer_3[0], RRDB_layer_2[0], RRDB_layer_1[0], inputs1], concat_dim=3, name='RRDB_layer_1_c1_{}_{}'.format(i, 3))  #
        RRDB_layer_3_c1 = Conv2d(RRDB_layer_3_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='RCB_conv1_{}_{}'.format(i, 3))  #

        RRDB_layer_3_c1 = ElementwiseLayer([RRDB_layer_3_c1, inputs1], tf.add, name='RCB_conv1_final_{}_{}'.format(i, 3))

        return RRDB_layer_3_c1, RRDB_layer_3[1]

def RRDB_Cascading_1_SFT_ResNet(layer1, layer2, C_nums=4, reuse=False):
    """ Generator for image SR
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    inputs1 = layer1
    i = layer2
    w_init = tf.variance_scaling_initializer(scale=0.01, seed=7)
    channel = inputs1.outputs.get_shape()[-1]
    with tf.variable_scope("RRDB_Cascading_1_SFT_ResNet", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        RRDB_layer_1 = Dense_Block_SFT_ResNet(inputs1, i, 1)
        RRDB_layer_1_c1 = ConcatLayer([RRDB_layer_1[0], inputs1], concat_dim=3, name='RRDB_layer_1_c1_{}_{}'.format(i, 1))  #
        RRDB_layer_1_c1 = Conv2d(RRDB_layer_1_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='RCB_conv1_{}_{}'.format(i, 1))  #

        RRDB_layer_2 = Dense_Block_Rethinking_SFT_ResNet(RRDB_layer_1_c1, RRDB_layer_1[1], i, 2)  #
        RRDB_layer_2_c1 = ConcatLayer([RRDB_layer_2[0], RRDB_layer_1[0], inputs1], concat_dim=3, name='RRDB_layer_1_c1_{}_{}'.format(i, 2))  #
        RRDB_layer_2_c1 = Conv2d(RRDB_layer_2_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='RCB_conv1_{}_{}'.format(i, 2))  #

        RRDB_layer_3 = Dense_Block_Rethinking_SFT_ResNet(RRDB_layer_2_c1, RRDB_layer_2[1], i, 3)  #
        RRDB_layer_3_c1 = ConcatLayer([RRDB_layer_3[0], RRDB_layer_2[0], RRDB_layer_1[0], inputs1], concat_dim=3, name='RRDB_layer_1_c1_{}_{}'.format(i, 3))  #
        RRDB_layer_3_c1 = Conv2d(RRDB_layer_3_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='RCB_conv1_{}_{}'.format(i, 3))  #

        RRDB_layer_3_c1 = ElementwiseLayer([RRDB_layer_3_c1, inputs1], tf.add, name='RCB_conv1_final_{}_{}'.format(i, 3))

        return RRDB_layer_3_c1, RRDB_layer_3[1]

def RRDB_ShareSkip_2(layer1, layer2, layer3, C_nums=4, reuse=False):
    """ Generator for image SR
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    inputs1 = layer1
    inputs2 = layer2
    i = layer3
    w_init = tf.variance_scaling_initializer(scale=0.01, seed=7)
    channel = inputs1.outputs.get_shape()[-1]
    with tf.variable_scope("RRDB_ShareSkip_2", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        for j in range(3):
            if j == 0:
                x = Dense_Block_Rethinking_ResNet(inputs1, inputs2, i, j)  #
            else:
                x = Dense_Block_Rethinking_ResNet(x[0], x[1], i, j)  #
            if j <= 1:
                x = list(x)
                x[0] = ElementwiseLayer([x[0], inputs1], tf.add, name='RRDB_share_skip2_add_{}_{}'.format(i, j))  # Share-source Skip Connections
                x = tuple(x)

        x[0].outputs = tf.multiply(x[0].outputs, 0.2)  # residual scale

        x = list(x)
        x[0] = ElementwiseLayer([x[0], inputs1], tf.add, name='RCB_conv2_final_{}_{}'.format(i, 3))
        x = tuple(x)

        return x[0], x[1]

def RRDB_ShareSkip_2_SFTResNet(layer1, layer2, layer3, C_nums=4, reuse=False):
    """ Generator for image SR
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    inputs1 = layer1
    inputs2 = layer2
    i = layer3
    w_init = tf.variance_scaling_initializer(scale=0.01, seed=7)
    channel = inputs1.outputs.get_shape()[-1]
    with tf.variable_scope("RRDB_ShareSkip_2_SFTResNet", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        for j in range(3):
            if j == 0:
                x = Dense_Block_Rethinking_SFT_ResNet(inputs1, inputs2, i, j)  #
            else:
                x = Dense_Block_Rethinking_SFT_ResNet(x[0], x[1], i, j)  #
            if j <= 1:
                x = list(x)
                x[0] = ElementwiseLayer([x[0], inputs1], tf.add, name='RRDB_share_skip2_SFTResNet_add_{}_{}'.format(i, j))  # Share-source Skip Connections
                x = tuple(x)

        x[0].outputs = tf.multiply(x[0].outputs, 0.2)  # residual scale

        x = list(x)
        x[0] = ElementwiseLayer([x[0], inputs1], tf.add, name='RCB_conv2_SFTResNet_final_{}_{}'.format(i, 3))
        x = tuple(x)

        return x[0], x[1]

def RRDB_Cascading_2(layer1, layer2, layer3, C_nums=4, reuse=False):
    """ Generator for image SR
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    inputs1 = layer1
    inputs2 = layer2
    i = layer3
    w_init = tf.variance_scaling_initializer(scale=0.01, seed=7)
    channel = inputs1.outputs.get_shape()[-1]
    with tf.variable_scope("RRDB_Cascading_2", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        RRDB_layer2_1 = Dense_Block_Rethinking(inputs1, inputs2, i, 1)
        RRDB_layer2_1_c1 = ConcatLayer([RRDB_layer2_1[0], inputs1], concat_dim=3, name='RRDB_layer2_1_c1_{}_{}'.format(i, 1))  #
        RRDB_layer2_1_c1 = Conv2d(RRDB_layer2_1_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='RCB2_conv1_{}_{}'.format(i, 1))  #

        RRDB_layer2_2 = Dense_Block_Rethinking(RRDB_layer2_1_c1, RRDB_layer2_1[1], i, 2)  #
        RRDB_layer2_2_c1 = ConcatLayer([RRDB_layer2_2[0], RRDB_layer2_1[0], inputs1], concat_dim=3, name='RRDB_layer2_1_c1_{}_{}'.format(i, 2))  #
        RRDB_layer2_2_c1 = Conv2d(RRDB_layer2_2_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='RCB2_conv1_{}_{}'.format(i, 2))  #

        RRDB_layer2_3 = Dense_Block_Rethinking(RRDB_layer2_2_c1, RRDB_layer2_2[1], i, 3)  #
        RRDB_layer2_3_c1 = ConcatLayer([RRDB_layer2_3[0], RRDB_layer2_2[0], RRDB_layer2_1[0], inputs1], concat_dim=3, name='RRDB_layer2_1_c1_{}_{}'.format(i, 3))  #
        RRDB_layer2_3_c1 = Conv2d(RRDB_layer2_3_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='RCB2_conv1_{}_{}'.format(i, 3))  #

        RRDB_layer2_3_c1 = ElementwiseLayer([RRDB_layer2_3_c1, inputs1], tf.add, name='RCB2_conv1_final_{}_{}'.format(i, 3))

        return RRDB_layer2_3_c1, RRDB_layer2_3[1]

def RRDB_Cascading_2_SFT(layer1, layer2, layer3, C_nums=4, reuse=False):
    """ Generator for image SR
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    inputs1 = layer1
    inputs2 = layer2
    i = layer3
    w_init = tf.variance_scaling_initializer(scale=0.01, seed=7)
    channel = inputs1.outputs.get_shape()[-1]
    with tf.variable_scope("RRDB_Cascading_2_SFT", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        RRDB_layer2_1 = Dense_Block_Rethinking_SFT(inputs1, inputs2, i, 1)
        RRDB_layer2_1_c1 = ConcatLayer([RRDB_layer2_1[0], inputs1], concat_dim=3, name='RRDB_layer2_1_c1_{}_{}'.format(i, 1))  #
        RRDB_layer2_1_c1 = Conv2d(RRDB_layer2_1_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='RCB2_conv1_{}_{}'.format(i, 1))  #

        RRDB_layer2_2 = Dense_Block_Rethinking_SFT(RRDB_layer2_1_c1, RRDB_layer2_1[1], i, 2)  #
        RRDB_layer2_2_c1 = ConcatLayer([RRDB_layer2_2[0], RRDB_layer2_1[0], inputs1], concat_dim=3, name='RRDB_layer2_1_c1_{}_{}'.format(i, 2))  #
        RRDB_layer2_2_c1 = Conv2d(RRDB_layer2_2_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='RCB2_conv1_{}_{}'.format(i, 2))  #

        RRDB_layer2_3 = Dense_Block_Rethinking_SFT(RRDB_layer2_2_c1, RRDB_layer2_2[1], i, 3)  #
        RRDB_layer2_3_c1 = ConcatLayer([RRDB_layer2_3[0], RRDB_layer2_2[0], RRDB_layer2_1[0], inputs1], concat_dim=3, name='RRDB_layer2_1_c1_{}_{}'.format(i, 3))  #
        RRDB_layer2_3_c1 = Conv2d(RRDB_layer2_3_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='RCB2_conv1_{}_{}'.format(i, 3))  #

        RRDB_layer2_3_c1 = ElementwiseLayer([RRDB_layer2_3_c1, inputs1], tf.add, name='RCB2_conv1_final_{}_{}'.format(i, 3))

        return RRDB_layer2_3_c1, RRDB_layer2_3[1]

def RRDB_Cascading_2_SFT_ResNet(layer1, layer2, layer3, C_nums=4, reuse=False):
    """ Generator for image SR
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    inputs1 = layer1
    inputs2 = layer2
    i = layer3
    w_init = tf.variance_scaling_initializer(scale=0.01, seed=7)
    channel = inputs1.outputs.get_shape()[-1]
    with tf.variable_scope("RRDB_Cascading_2_SFT_ResNet", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        RRDB_layer2_1 = Dense_Block_Rethinking_SFT_ResNet(inputs1, inputs2, i, 1)
        RRDB_layer2_1_c1 = ConcatLayer([RRDB_layer2_1[0], inputs1], concat_dim=3, name='RRDB_layer2_1_c1_{}_{}'.format(i, 1))  #
        RRDB_layer2_1_c1 = Conv2d(RRDB_layer2_1_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='RCB2_conv1_{}_{}'.format(i, 1))  #

        RRDB_layer2_2 = Dense_Block_Rethinking_SFT_ResNet(RRDB_layer2_1_c1, RRDB_layer2_1[1], i, 2)  #
        RRDB_layer2_2_c1 = ConcatLayer([RRDB_layer2_2[0], RRDB_layer2_1[0], inputs1], concat_dim=3, name='RRDB_layer2_1_c1_{}_{}'.format(i, 2))  #
        RRDB_layer2_2_c1 = Conv2d(RRDB_layer2_2_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='RCB2_conv1_{}_{}'.format(i, 2))  #

        RRDB_layer2_3 = Dense_Block_Rethinking_SFT_ResNet(RRDB_layer2_2_c1, RRDB_layer2_2[1], i, 3)  #
        RRDB_layer2_3_c1 = ConcatLayer([RRDB_layer2_3[0], RRDB_layer2_2[0], RRDB_layer2_1[0], inputs1], concat_dim=3, name='RRDB_layer2_1_c1_{}_{}'.format(i, 3))  #
        RRDB_layer2_3_c1 = Conv2d(RRDB_layer2_3_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='RCB2_conv1_{}_{}'.format(i, 3))  #

        RRDB_layer2_3_c1 = ElementwiseLayer([RRDB_layer2_3_c1, inputs1], tf.add, name='RCB2_conv1_final_{}_{}'.format(i, 3))

        return RRDB_layer2_3_c1, RRDB_layer2_3[1]

def SFT(layer1, layer2, layer3, C_nums=4, reuse=False):
    """ Generator for image SR
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    inputs1 = layer1
    i = layer2
    j = layer3
    w_init = tf.variance_scaling_initializer(scale=0.01, seed=7)
    channel = inputs1.outputs.get_shape()[-1]
    with tf.variable_scope("SFT", reuse=reuse) as vs:
        alpha_conv1 = Conv2d(inputs1, channel, (5, 5), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='alpha_conv1_{}_{}'.format(i, j))  #
        alpha_conv1_pool = PoolLayer(alpha_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool = tf.nn.avg_pool, name='alpha_conv1_pool')

        alpha_conv2 = Conv2d(alpha_conv1_pool, int(channel*2), (5, 5), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='alpha_conv2_{}_{}'.format(i, j))  #
        alpha_conv2_pool = PoolLayer(alpha_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.avg_pool, name='alpha_conv2_pool')

        alpha_conv3 = Conv2d(alpha_conv2_pool, int(channel*4), (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='alpha_conv3_{}_{}'.format(i, j))  #
        alpha_conv3_pool = PoolLayer(alpha_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.avg_pool, name='alpha_conv3_pool')

        alpha_conv4 = Conv2d(alpha_conv3_pool, int(channel*4), (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='alpha_conv4_{}_{}'.format(i, j))  #
        alpha_conv4_up = UpSampling2dLayer(alpha_conv4, size=[2 * alpha_conv4.outputs.shape[1], 2 * alpha_conv4.outputs.shape[2]], is_scale=False, method=0, name='alpha_conv4_up_{}_{}'.format(i, j))
        alpha_conv4_up = ConcatLayer([alpha_conv4_up, alpha_conv3], concat_dim=3, name='alpha_conv4_up_concat_{}_{}'.format(i, j))  #

        alpha_conv5 = Conv2d(alpha_conv4_up, int(channel*4), (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='alpha_conv5_{}_{}'.format(i, j))  #
        alpha_conv5_up = UpSampling2dLayer(alpha_conv5, size=[2 * alpha_conv5.outputs.shape[1], 2 * alpha_conv5.outputs.shape[2]], is_scale=False, method=0, name='alpha_conv5_up_{}_{}'.format(i, j))
        alpha_conv5_up = ConcatLayer([alpha_conv5_up, alpha_conv2], concat_dim=3, name='alpha_conv5_up_concat_{}_{}'.format(i, j))  #

        alpha_conv6 = Conv2d(alpha_conv5_up, int(channel*2), (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='alpha_conv6_{}_{}'.format(i, j))  #
        alpha_conv6_up = UpSampling2dLayer(alpha_conv6, size=[2 * alpha_conv6.outputs.shape[1], 2 * alpha_conv6.outputs.shape[2]], is_scale=False, method=0, name='alpha_conv6_up_{}_{}'.format(i, j))
        alpha_conv6_up = ConcatLayer([alpha_conv6_up, alpha_conv1], concat_dim=3, name='alpha_conv6_up_concat_{}_{}'.format(i, j))  #

        alpha_conv6_up = Conv2d(alpha_conv6_up, channel, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='alpha_final_{}_{}'.format(i, j))  #

        ########################################################################################################################################################################################

        beta_conv1 = Conv2d(inputs1, channel, (5, 5), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='beta_conv1_{}_{}'.format(i, j))  #
        beta_conv1_pool = PoolLayer(beta_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.avg_pool, name='beta_conv1_pool')

        beta_conv2 = Conv2d(beta_conv1_pool, int(channel*2), (5, 5), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='beta_conv2_{}_{}'.format(i, j))  #
        beta_conv2_pool = PoolLayer(beta_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.avg_pool, name='beta_conv2_pool')

        beta_conv3 = Conv2d(beta_conv2_pool, int(channel*4), (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='beta_conv3_{}_{}'.format(i, j))  #
        beta_conv3_pool = PoolLayer(beta_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.avg_pool, name='beta_conv3_pool')

        beta_conv4 = Conv2d(beta_conv3_pool, int(channel*4), (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='beta_conv4_{}_{}'.format(i, j))  #
        beta_conv4_up = UpSampling2dLayer(beta_conv4, size=[2 * beta_conv4.outputs.shape[1], 2 * beta_conv4.outputs.shape[2]], is_scale=False, method=0, name='beta_conv4_up_{}_{}'.format(i, j))
        beta_conv4_up = ConcatLayer([beta_conv4_up, beta_conv3], concat_dim=3, name='beta_conv4_up_concat_{}_{}'.format(i, j))  #

        beta_conv5 = Conv2d(beta_conv4_up, int(channel*4), (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='beta_conv5_{}_{}'.format(i, j))  #
        beta_conv5_up = UpSampling2dLayer(beta_conv5, size=[2 * beta_conv5.outputs.shape[1], 2 * beta_conv5.outputs.shape[2]], is_scale=False, method=0, name='beta_conv5_up_{}_{}'.format(i, j))
        beta_conv5_up = ConcatLayer([beta_conv5_up, beta_conv2], concat_dim=3, name='beta_conv5_up_concat_{}_{}'.format(i, j))  #

        beta_conv6 = Conv2d(beta_conv5_up, int(channel*2), (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='beta_conv6_{}_{}'.format(i, j))  #
        beta_conv6_up = UpSampling2dLayer(beta_conv6, size=[2 * beta_conv6.outputs.shape[1], 2 * beta_conv6.outputs.shape[2]], is_scale=False, method=0, name='beta_conv6_up_{}_{}'.format(i, j))
        beta_conv6_up = ConcatLayer([beta_conv6_up, beta_conv1], concat_dim=3, name='beta_conv6_up_concat_{}_{}'.format(i, j))  #

        beta_conv6_up = Conv2d(beta_conv6_up, channel, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='beta_final_{}_{}'.format(i, j))  #

        #############################################################################################################################################################################
        inputs1 = ElementwiseLayer([inputs1, alpha_conv6_up], tf.multiply, name='SFT_alpha_{}_{}'.format(i, j))
        inputs1 = ElementwiseLayer([inputs1, beta_conv6_up], tf.add, name='SFT_alpha_beta_{}_{}'.format(i, j))
        inputs1 = Conv2d(inputs1, 32, (1, 1), (1, 1), padding='SAME', act=tf.nn.leaky_relu, name='SFT_final_{}_{}'.format(i, j))

        return inputs1

def SFT_ResNet(layer1, layer2, layer3, C_nums=4, reuse=False):
    """ Generator for image SR
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    #n = layer1
    #m = layer1
    inputs1 = layer1
    i = layer2
    j = layer3
    w_init = tf.variance_scaling_initializer(scale=0.01, seed=7)
    channel = inputs1.outputs.get_shape()[-1]
    with tf.variable_scope("SFT_ResNet", reuse=reuse) as vs:
        for k in range(1):
            nn = Conv2d(inputs1, channel, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='alpha_conv1_{}_{}_{}'.format(i, j, k))
            nn = Conv2d(nn, channel, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='alpha_conv2_{}_{}_{}'.format(i, j, k))

            nn = ElementwiseLayer([inputs1, nn], tf.add, name='alpha_conv3_{}_{}_{}'.format(i, j, k))
            #n = nn

        #n = ElementwiseLayer([n, inputs1], tf.add, name='alpha_final_{}_{}'.format(i, j))

        for t in range(1):
            mm = Conv2d(inputs1, channel, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='beta_conv1_{}_{}_{}'.format(i, j, t))
            mm = Conv2d(mm, channel, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='beta_conv2_{}_{}_{}'.format(i, j, t))

            mm = ElementwiseLayer([inputs1, mm], tf.add, name='beta_conv3_{}_{}_{}'.format(i, j, t))
            #m = mm

        #m = ElementwiseLayer([m, inputs1], tf.add, name='beta_final_{}_{}'.format(i, j))

        inputs1 = ElementwiseLayer([inputs1, nn], tf.multiply, name='SFT_alpha_{}_{}'.format(i, j))
        inputs1 = ElementwiseLayer([inputs1, mm], tf.add, name='SFT_alpha_beta_{}_{}'.format(i, j))
        inputs1 = Conv2d(inputs1, 32, (1, 1), (1, 1), padding='SAME', act=tf.nn.leaky_relu, name='SFT_final_{}_{}'.format(i, j))

        return inputs1

def ESRGAN_OutCascading_InShareSkip_Rethinking_efficient_g(t_image, t_image_up, is_train=False, input_depth=1, n_layers=14, n_sub_layers=3, BN=False, ratio=1, reuse=False):
    """ Generator for image SR
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    w_init = tf.variance_scaling_initializer(scale=0.01, seed=7)
    #w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("ESRGAN_OutCascading_InShareSkip_Rethinking_efficient_g", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)

        n = InputLayer(t_image, name='in')

        n_up = InputLayer(t_image_up, name='in_up')

        n_1 = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='n_1')
        #temp = n_1
        #rgb_in = n_1
        channel = n_1.outputs.get_shape()[-1]

        Out_RCB_layer_1 = RRDB_ShareSkip_1(n_1, 1)  # i=1
        Out_RCB_layer_1_c1 = ConcatLayer([Out_RCB_layer_1[0], n_1], concat_dim=3, name='Out_RCB_layer_1_c1_{}'.format(1))  # i=1
        Out_RCB_layer_1_c1 = Conv2d(Out_RCB_layer_1_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='Out_RCB_conv1_{}'.format(1))  # i=1

        Out_RCB_layer_2 = RRDB_ShareSkip_2(Out_RCB_layer_1_c1, Out_RCB_layer_1[1], 2)  # i=1
        Out_RCB_layer_2_c1 = ConcatLayer([Out_RCB_layer_2[0], Out_RCB_layer_1[0], n_1], concat_dim=3, name='Out_RCB_layer_1_c1_{}'.format(2))  # i=2
        Out_RCB_layer_2_c1 = Conv2d(Out_RCB_layer_2_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='Out_RCB_conv1_{}'.format(2))  # i=

        Out_RCB_layer_3 = RRDB_ShareSkip_2(Out_RCB_layer_2_c1, Out_RCB_layer_2[1], 3)  # i=1
        Out_RCB_layer_3_c1 = ConcatLayer([Out_RCB_layer_3[0], Out_RCB_layer_2[0], Out_RCB_layer_1[0], n_1], concat_dim=3, name='Out_RCB_layer_1_c1_{}'.format(3))  # i=2
        Out_RCB_layer_3_c1 = Conv2d(Out_RCB_layer_3_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='Out_RCB_conv1_{}'.format(3))  #

        Out_RCB_layer_4 = RRDB_ShareSkip_2(Out_RCB_layer_3_c1, Out_RCB_layer_3[1], 4)  # i=1
        Out_RCB_layer_4_c1 = ConcatLayer([Out_RCB_layer_4[0], Out_RCB_layer_3[0], Out_RCB_layer_2[0], Out_RCB_layer_1[0], n_1], concat_dim=3, name='Out_RCB_layer_1_c1_{}'.format(4))  # i=2
        Out_RCB_layer_4_c1 = Conv2d(Out_RCB_layer_4_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='Out_RCB_conv1_{}'.format(4))  #

        #Out_RCB_layer_5 = RRDB_ShareSkip_2(Out_RCB_layer_4_c1, Out_RCB_layer_4[1], 5)  # i=1
        #Out_RCB_layer_5_c1 = ConcatLayer([Out_RCB_layer_5[0], Out_RCB_layer_4[0], Out_RCB_layer_3[0], Out_RCB_layer_2[0], Out_RCB_layer_1[0], n_1], concat_dim=3, name='Out_RCB_layer_1_c1_{}'.format(5))  # i=2
        #Out_RCB_layer_5_c1 = Conv2d(Out_RCB_layer_5_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='Out_RCB_conv1_{}'.format(5))  #

        #Out_RCB_layer_6 = RRDB_ShareSkip_2(Out_RCB_layer_5_c1, Out_RCB_layer_5[1], 6)  # i=1
        #Out_RCB_layer_6_c1 = ConcatLayer([Out_RCB_layer_6[0], Out_RCB_layer_5[0], Out_RCB_layer_4[0], Out_RCB_layer_3[0], Out_RCB_layer_2[0], Out_RCB_layer_1[0], n_1], concat_dim=3, name='Out_RCB_layer_1_c1_{}'.format(6))  # i=2
        #Out_RCB_layer_6_c1 = Conv2d(Out_RCB_layer_6_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='Out_RCB_conv1_{}'.format(6))  #

        #Out_RCB_layer_7 = RRDB_ShareSkip_2(Out_RCB_layer_6_c1, Out_RCB_layer_6[1], 7)  # i=1
        #Out_RCB_layer_7_c1 = ConcatLayer([Out_RCB_layer_7[0], Out_RCB_layer_6[0], Out_RCB_layer_5[0], Out_RCB_layer_4[0], Out_RCB_layer_3[0], Out_RCB_layer_2[0], Out_RCB_layer_1[0], n_1], concat_dim=3, name='Out_RCB_layer_1_c1_{}'.format(7))  # i=2
        #Out_RCB_layer_7_c1 = Conv2d(Out_RCB_layer_7_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='Out_RCB_conv1_{}'.format(7))  #

        #Out_RCB_layer_8 = RRDB_ShareSkip_2(Out_RCB_layer_7_c1, Out_RCB_layer_7[1], 8)  # i=1
        #Out_RCB_layer_8_c1 = ConcatLayer([Out_RCB_layer_8[0], Out_RCB_layer_7[0], Out_RCB_layer_6[0], Out_RCB_layer_5[0], Out_RCB_layer_4[0], Out_RCB_layer_3[0], Out_RCB_layer_2[0], Out_RCB_layer_1[0], n_1], concat_dim=3, name='Out_RCB_layer_1_c1_{}'.format(8))  # i=2
        #Out_RCB_layer_8_c1 = Conv2d(Out_RCB_layer_8_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='Out_RCB_conv1_{}'.format(8))  #

        #Out_RCB_layer_9 = RRDB_ShareSkip_2(Out_RCB_layer_8_c1, Out_RCB_layer_8[1], 9)  # i=1
        #Out_RCB_layer_9_c1 = ConcatLayer([Out_RCB_layer_9[0], Out_RCB_layer_8[0], Out_RCB_layer_7[0], Out_RCB_layer_6[0], Out_RCB_layer_5[0], Out_RCB_layer_4[0], Out_RCB_layer_3[0], Out_RCB_layer_2[0], Out_RCB_layer_1[0], n_1], concat_dim=3, name='Out_RCB_layer_1_c1_{}'.format(9))  # i=2
        #Out_RCB_layer_9_c1 = Conv2d(Out_RCB_layer_9_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='Out_RCB_conv1_{}'.format(9))  #

        #Out_RCB_layer_10 = RRDB_ShareSkip_2(Out_RCB_layer_9_c1, Out_RCB_layer_9[1], 10)  # i=1
        #Out_RCB_layer_10_c1 = ConcatLayer([Out_RCB_layer_10[0], Out_RCB_layer_9[0], Out_RCB_layer_8[0], Out_RCB_layer_7[0], Out_RCB_layer_6[0], Out_RCB_layer_5[0], Out_RCB_layer_4[0], Out_RCB_layer_3[0], Out_RCB_layer_2[0], Out_RCB_layer_1[0], n_1], concat_dim=3, name='Out_RCB_layer_1_c1_{}'.format(10))  # i=2
        #Out_RCB_layer_10_c1 = Conv2d(Out_RCB_layer_10_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='Out_RCB_conv1_{}'.format(10))  #

        #Out_RCB_layer_11 = RRDB_ShareSkip_2(Out_RCB_layer_10_c1, Out_RCB_layer_10[1], 11)  # i=1
        #Out_RCB_layer_11_c1 = ConcatLayer([Out_RCB_layer_11[0], Out_RCB_layer_10[0], Out_RCB_layer_9[0], Out_RCB_layer_8[0], Out_RCB_layer_7[0], Out_RCB_layer_6[0], Out_RCB_layer_5[0], Out_RCB_layer_4[0], Out_RCB_layer_3[0], Out_RCB_layer_2[0], Out_RCB_layer_1[0], n_1], concat_dim=3, name='Out_RCB_layer_1_c1_{}'.format(11))  # i=2
        #Out_RCB_layer_11_c1 = Conv2d(Out_RCB_layer_11_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='Out_RCB_conv1_{}'.format(11))  #

        #Out_RCB_layer_12 = RRDB_ShareSkip_2(Out_RCB_layer_11_c1, Out_RCB_layer_11[1], 12)  # i=1
        #Out_RCB_layer_12_c1 = ConcatLayer([Out_RCB_layer_12[0], Out_RCB_layer_11[0], Out_RCB_layer_10[0], Out_RCB_layer_9[0], Out_RCB_layer_8[0], Out_RCB_layer_7[0], Out_RCB_layer_6[0], Out_RCB_layer_5[0], Out_RCB_layer_4[0], Out_RCB_layer_3[0], Out_RCB_layer_2[0], Out_RCB_layer_1[0], n_1], concat_dim=3, name='Out_RCB_layer_1_c1_{}'.format(12))  # i=2
        #Out_RCB_layer_12_c1 = Conv2d(Out_RCB_layer_12_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='Out_RCB_conv1_{}'.format(12))  #

        #Out_RCB_layer_13 = RRDB_ShareSkip_2(Out_RCB_layer_12_c1, Out_RCB_layer_12[1], 13)  # i=1
        #Out_RCB_layer_13_c1 = ConcatLayer([Out_RCB_layer_13[0], Out_RCB_layer_12[0], Out_RCB_layer_11[0], Out_RCB_layer_10[0], Out_RCB_layer_9[0], Out_RCB_layer_8[0], Out_RCB_layer_7[0], Out_RCB_layer_6[0], Out_RCB_layer_5[0], Out_RCB_layer_4[0], Out_RCB_layer_3[0], Out_RCB_layer_2[0], Out_RCB_layer_1[0], n_1], concat_dim=3, name='Out_RCB_layer_1_c1_{}'.format(13))  # i=2
        #Out_RCB_layer_13_c1 = Conv2d(Out_RCB_layer_13_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='Out_RCB_conv1_{}'.format(13))  #

        #Out_RCB_layer_14 = RRDB_ShareSkip_2(Out_RCB_layer_13_c1, Out_RCB_layer_13[1], 14)  # i=1
        #Out_RCB_layer_14_c1 = ConcatLayer([Out_RCB_layer_14[0], Out_RCB_layer_13[0], Out_RCB_layer_12[0], Out_RCB_layer_11[0], Out_RCB_layer_10[0], Out_RCB_layer_9[0], Out_RCB_layer_8[0], Out_RCB_layer_7[0], Out_RCB_layer_6[0], Out_RCB_layer_5[0], Out_RCB_layer_4[0], Out_RCB_layer_3[0], Out_RCB_layer_2[0], Out_RCB_layer_1[0], n_1], concat_dim=3, name='Out_RCB_layer_1_c1_{}'.format(14))  # i=2
        #Out_RCB_layer_14_c1 = Conv2d(Out_RCB_layer_14_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='Out_RCB_conv1_{}'.format(14))  #

        #Out_RCB_layer_15 = RRDB_ShareSkip_2(Out_RCB_layer_14_c1, Out_RCB_layer_14[1], 15)  # i=1
        #Out_RCB_layer_15_c1 = ConcatLayer([Out_RCB_layer_15[0], Out_RCB_layer_14[0], Out_RCB_layer_13[0], Out_RCB_layer_12[0], Out_RCB_layer_11[0], Out_RCB_layer_10[0], Out_RCB_layer_9[0], Out_RCB_layer_8[0], Out_RCB_layer_7[0], Out_RCB_layer_6[0], Out_RCB_layer_5[0], Out_RCB_layer_4[0], Out_RCB_layer_3[0], Out_RCB_layer_2[0], Out_RCB_layer_1[0], n_1], concat_dim=3, name='Out_RCB_layer_1_c1_{}'.format(15))  # i=2
        #Out_RCB_layer_15_c1 = Conv2d(Out_RCB_layer_15_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='Out_RCB_conv1_{}'.format(15))  #

        #Out_RCB_layer_16 = RRDB_ShareSkip_2(Out_RCB_layer_15_c1, Out_RCB_layer_15[1], 16)  # i=1
        #Out_RCB_layer_16_c1 = ConcatLayer([Out_RCB_layer_16[0], Out_RCB_layer_15[0], Out_RCB_layer_14[0], Out_RCB_layer_13[0], Out_RCB_layer_12[0], Out_RCB_layer_11[0], Out_RCB_layer_10[0], Out_RCB_layer_9[0], Out_RCB_layer_8[0], Out_RCB_layer_7[0], Out_RCB_layer_6[0], Out_RCB_layer_5[0], Out_RCB_layer_4[0], Out_RCB_layer_3[0], Out_RCB_layer_2[0], Out_RCB_layer_1[0], n_1], concat_dim=3, name='Out_RCB_layer_1_c1_{}'.format(16))  # i=2
        #Out_RCB_layer_16_c1 = Conv2d(Out_RCB_layer_16_c1, channel, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='Out_RCB_conv1_{}'.format(16))  #

        Out_RCB_layer_4_c1 = Conv2d(Out_RCB_layer_4_c1, 64, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='RGB_concat_final_conv2')
        Out_RCB_layer_4_c1 = ElementwiseLayer([Out_RCB_layer_4_c1, n_1], tf.add, 'add1')
        # B residual blocks end

        Out_RCB_layer_4_c1 = Conv2d(Out_RCB_layer_4_c1, 64, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, name='out1')
        Out_RCB_layer_4_c1 = Conv2d(Out_RCB_layer_4_c1, 3, (3, 3), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out2')
        Out_RCB_layer_4_c1 = ElementwiseLayer([Out_RCB_layer_4_c1, n_up], tf.add, 'add2')

        return Out_RCB_layer_4_c1


def SRGAN_g(t_image, is_train=False, input_depth=1, n_layers=16, BN=False, ratio=2, reuse=False):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    w_init = tf.variance_scaling_initializer(scale=0.01, seed=7)
    #w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)

        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (9, 9), (1, 1), act=None, padding='SAME', W_init=w_init, name='n64s1/c')
        n = PReluLayer(n, name='pr/c')
        temp = n

        # B residual blocks
        for i in range(n_layers):
            if BN:
                nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
                nn = BatchNormLayer(nn, act=tf.identity, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
                nn = PReluLayer(nn, name='pr/c1/%s' % i)
                nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
                nn = BatchNormLayer(nn, act=tf.identity, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            else:
                nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
                nn = PReluLayer(nn, name='pr/c1/%s' % i)
                nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)

            nn = ElementwiseLayer([n, nn], tf.add, 'b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        if BN:
            n = BatchNormLayer(n, act=tf.identity, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, 'add3')
        # B residual blocks end

        if ratio==1.5:
            n = Conv2d(n, 288, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n288s1/1') # changed this from 576 (64*9) to 288 (32*9)
            n = SubpixelConv2d(n, scale=3, n_out_channel=None, act=tf.identity, name='pixelshufflerx3/1')
            n = Conv2d(n, 32, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, name='n32s2')
            n = PReluLayer(n, name='pr/pixelshufflerx3/1')

        if ratio==3:
            n = Conv2d(n, 288, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
            n = SubpixelConv2d(n, scale=3, n_out_channel=None, act=tf.identity, name='pixelshufflerx3/1')
            n = PReluLayer(n, name='pr/pixelshufflerx3/1')

        if ratio==2 or ratio==4:
            n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
            n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.identity, name='pixelshufflerx2/1')
            n = PReluLayer(n, name='pr/pixelshufflerx2/1')

        if ratio==4:
            n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
            n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.identity, name='pixelshufflerx2/2')
            n = PReluLayer(n, name='pr/pixelshufflerx2/2')

        n = Conv2d(n, input_depth, (9, 9), (1, 1), act=tf.identity, padding='SAME', W_init=w_init, name='out')

        return n

def SRGAN_g_new(t_image, t_image_up, is_train=False, input_depth=1, n_layers=16, BN=False, ratio=2, reuse=False):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    w_init = tf.variance_scaling_initializer(scale=0.01, seed=7)
    #w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("SRGAN_g_new", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)

        n = InputLayer(t_image, name='in')

        n_up = InputLayer(t_image_up, name='in_up')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n64s1/c')
        n = PReluLayer(n, name='pr/c')
        temp = n

        # B residual blocks
        for i in range(n_layers):
            if BN:
                nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
                nn = BatchNormLayer(nn, act=tf.identity, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
                nn = PReluLayer(nn, name='pr/c1/%s' % i)
                nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
                nn = BatchNormLayer(nn, act=tf.identity, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            else:
                nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
                nn = PReluLayer(nn, name='pr/c1/%s' % i)
                nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)

            nn = ElementwiseLayer([n, nn], tf.add, 'b_residual_add/%s' % i)
            n = nn

        n = ElementwiseLayer([n, temp], tf.add, 'add1')
        # B residual blocks end

        if ratio==1.5:
            n = Conv2d(n, 576, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n288s1/1') # changed this from 576 (64*9) to 288 (32*9)
            n = SubpixelConv2d(n, scale=3, n_out_channel=None, act=tf.identity, name='pixelshufflerx3/1')
            n = PReluLayer(n, name='pr/pixelshufflerx3/1')
            n = Conv2d(n, 64, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, name='n32s2')
            n = PReluLayer(n, name='pr/pixelshufflerx3/2')

        if ratio==3:
            n = Conv2d(n, 576, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
            n = SubpixelConv2d(n, scale=3, n_out_channel=None, act=tf.identity, name='pixelshufflerx3/1')
            n = PReluLayer(n, name='pr/pixelshufflerx3/1')

        if ratio==2 or ratio==4:
            n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
            n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.identity, name='pixelshufflerx2/1')
            n = PReluLayer(n, name='pr/pixelshufflerx2/1')

        if ratio==4:
            n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
            n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.identity, name='pixelshufflerx2/2')
            n = PReluLayer(n, name='pr/pixelshufflerx2/2')

        n = Conv2d(n, input_depth, (3, 3), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')

        n = ElementwiseLayer([n, n_up], tf.add, 'add2')

        return n

def SRGAN_g_model1(t_image, is_train=False, input_depth=1, n_layers=16, BN=False, ratio=2, reuse=False):
    """ Generator similar to the one in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("SRGAN_g_model1", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n

        # B residual blocks
        for i in range(n_layers):
            if BN:
                nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
                nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
                nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
                nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            else:
                nn = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
                nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)

            nn = ElementwiseLayer([n, nn], tf.add, 'b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        if BN:
            n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, 'add3')
        # B residual blocks end

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')

        if ratio==4:
            n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
            n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')

        n = Conv2d(n, input_depth, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')

    return n

def SRGAN_g2(t_image, is_train=False, reuse=False):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)

    96x96 --> 384x384

    Use Resize Conv
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)

    size = t_image.get_shape().as_list()

    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n

        # B residual blocks
        for i in range(16):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, 'b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, 'add3')
        # B residual blacks end

        # n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
        # n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')
        #
        # n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
        # n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')

        ## 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
        n = UpSampling2dLayer(n, size=[size[1]*2, size[2]*2], is_scale=False, method=1, align_corners=False, name='up1/upsample2d')
        n = Conv2d(n, 64, (3, 3), (1, 1),
               padding='SAME', W_init=w_init, b_init=b_init, name='up1/conv2d')   # <-- may need to increase n_filter
        n = BatchNormLayer(n, act=tf.nn.relu,
                is_train=is_train, gamma_init=g_init, name='up1/batch_norm')

        n = UpSampling2dLayer(n, size=[size[1]*4, size[2]*4], is_scale=False, method=1, align_corners=False, name='up2/upsample2d')
        n = Conv2d(n, 32, (3, 3), (1, 1),
               padding='SAME', W_init=w_init, b_init=b_init, name='up2/conv2d')     # <-- may need to increase n_filter
        n = BatchNormLayer(n, act=tf.nn.relu,
                is_train=is_train, gamma_init=g_init, name='up2/batch_norm')

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n


def SRGAN_d2(t_image, is_train=False, reuse=False):
    """ Discriminator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    w_init = tf.variance_scaling_initializer(scale=0.01, seed=7)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x : tl.act.lrelu(x, 0.2)
    with tf.variable_scope("SRGAN_d2", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n64s1/c')

        n = Conv2d(n, 64, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s2/b')

        n = Conv2d(n, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s1/b')

        n = Conv2d(n, 128, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s2/b')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n256s1/b')

        n = Conv2d(n, 256, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n256s2/b')

        n = Conv2d(n, 512, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n512s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n512s1/b')

        n = Conv2d(n, 512, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n512s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n512s2/b')

        n = FlattenLayer(n, name='f')
        n = DenseLayer(n, n_units=1024, act=lrelu, name='d1024')
        n = DenseLayer(n, n_units=1, name='out')

        logits = n.outputs
        n.outputs = tf.nn.sigmoid(n.outputs)

        return n, logits, n.outputs

def SRGAN_d3(t_image, is_train=False, reuse=False):
    """ Discriminator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    # Mariana - Less powerfull discriminator compared to the original SRGAN_d2

    w_init = tf.variance_scaling_initializer(scale=0.01, seed=7)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x : tl.act.lrelu(x, 0.2)
    with tf.variable_scope("SRGAN_d3", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 32, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n32s1/c')

        n = Conv2d(n, 32, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n32s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n32s2/b')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b')

        n = Conv2d(n, 64, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s2/b')

        n = Conv2d(n, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s1/b')

        n = Conv2d(n, 128, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s2/b')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n256s1/b')

        n = Conv2d(n, 256, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n256s2/b')

        n = FlattenLayer(n, name='f')
        n = DenseLayer(n, n_units=512, act=lrelu, name='d512')
        n = DenseLayer(n, n_units=1, name='out')

        logits = n.outputs
        n.outputs = tf.nn.sigmoid(n.outputs)

        return n, logits, n.outputs

def SRGAN_d(input_images, is_train=True, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope("SRGAN_d", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lrelu,
                padding='SAME', W_init=w_init, name='h0/c')

        net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h1/c')
        net_h1 = BatchNormLayer(net_h1, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h1/bn')
        net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h2/c')
        net_h2 = BatchNormLayer(net_h2, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h2/bn')
        net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h3/c')
        net_h3 = BatchNormLayer(net_h3, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h3/bn')
        net_h4 = Conv2d(net_h3, df_dim*16, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h4/c')
        net_h4 = BatchNormLayer(net_h4, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h4/bn')
        net_h5 = Conv2d(net_h4, df_dim*32, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h5/c')
        net_h5 = BatchNormLayer(net_h5, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h5/bn')
        net_h6 = Conv2d(net_h5, df_dim*16, (1, 1), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h6/c')
        net_h6 = BatchNormLayer(net_h6, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h6/bn')
        net_h7 = Conv2d(net_h6, df_dim*8, (1, 1), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h7/c')
        net_h7 = BatchNormLayer(net_h7, is_train=is_train,
                gamma_init=gamma_init, name='h7/bn')

        net = Conv2d(net_h7, df_dim*2, (1, 1), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='res/c')
        net = BatchNormLayer(net, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='res/bn')
        net = Conv2d(net, df_dim*2, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='res/c2')
        net = BatchNormLayer(net, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='res/bn2')
        net = Conv2d(net, df_dim*8, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='res/c3')
        net = BatchNormLayer(net, is_train=is_train,
                gamma_init=gamma_init, name='res/bn3')
        net_h8 = ElementwiseLayer(layer=[net_h7, net],
                combine_fn=tf.add, name='res/add')
        net_h8.outputs = tl.act.lrelu(net_h8.outputs, 0.2)

        net_ho = FlattenLayer(net_h8, name='ho/flatten')
        net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity,
                W_init = w_init, name='ho/dense')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)

    return net_ho, logits, net_ho.outputs

def Vgg19_simple_api(rgb, reuse, convNum):
    """
    Build the VGG 19 Model

    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse) as vs:
        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else: # TF 1.0
            # print(rgb_scaled)
            red, green, blue = tf.split(rgb_scaled, 3, 3)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], axis=3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        """ input layer """
        net_in = InputLayer(bgr, name='input')
        """ conv1 """
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool1')
        """ conv2 """
        network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv2_2')
        if convNum == 2:
            conv = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool2')
        """ conv3 """
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_4')
        if convNum == 3:
            conv = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool3')
        """ conv4 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool4')                               # (batch_size, 14, 14, 512)
        if convNum == 0: # this was the original location
            conv = network
        """ conv5 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.identity,padding='SAME', name='conv5_4')
        # network = Conv2d(network, n_filter=512, filter_size=(3, 3),
        #            strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_4')
        if convNum == 6: # before the activation function (as in the ESRGAN paper)
            conv = network

        network.outputs = tf.nn.relu(network.outputs)

        if convNum == 5:
            conv = network

        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool5')                               # (batch_size, 7, 7, 512)
        """ fc 6~8 """
        network = FlattenLayer(network, name='flatten')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        print("build model finished: %fs" % (time.time() - start_time))
        return network, conv

# def vgg16_cnn_emb(t_image, reuse=False):
#     """ t_image = 244x244 [0~255] """
#     with tf.variable_scope("vgg16_cnn", reuse=reuse) as vs:
#         tl.layers.set_name_reuse(reuse)
#
#         mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
#         net_in = InputLayer(t_image - mean, name='vgg_input_im')
#         """ conv1 """
#         network = tl.layers.Conv2dLayer(net_in,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 3, 64],  # 64 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv1_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 64, 64],  # 64 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv1_2')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool1')
#         """ conv2 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 64, 128],  # 128 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv2_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 128, 128],  # 128 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv2_2')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool2')
#         """ conv3 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 128, 256],  # 256 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv3_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv3_2')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv3_3')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool3')
#         """ conv4 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 256, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv4_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv4_2')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv4_3')
#
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool4')
#         conv4 = network
#
#         """ conv5 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv5_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv5_2')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv5_3')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool5')
#
#         network = FlattenLayer(network, name='vgg_flatten')
#
#         # # network = DropoutLayer(network, keep=0.6, is_fix=True, is_train=is_train, name='vgg_out/drop1')
#         # new_network = tl.layers.DenseLayer(network, n_units=4096,
#         #                     act = tf.nn.relu,
#         #                     name = 'vgg_out/dense')
#         #
#         # # new_network = DropoutLayer(new_network, keep=0.8, is_fix=True, is_train=is_train, name='vgg_out/drop2')
#         # new_network = DenseLayer(new_network, z_dim, #num_lstm_units,
#         #             b_init=None, name='vgg_out/out')
#         return conv4, network
