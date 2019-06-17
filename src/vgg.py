# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import tensorflow as tf
import numpy as np
import scipy.io
import pdb

MEAN_PIXEL = np.array([ 123.68 ,  116.779,  103.939])

"""
重点是搞清楚这个是VGG16/VGG19，根据连接的基本结构判断，这个是读取的网络连接结构是VGG19.
VGG16: 2+2+3+3+3 + 3fc=16
VGG19: 2+2+4+4+4 + 3fc=19
每大层 对应的神经元的个数是相等的.
"""


def net(data_path, input_image):
    # 参数为：vgg19路径 + 图像矩阵.
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    # data = scipy.io.loadmat(data_path)
    # 深层次拷贝字典数据类型.
    data = data_path.copy()

    """
    个人认为多次读取路径的矩阵，明显的增加了I/O操作，可以只读取一次，然后之后直接使用.
    当然可能会重复的创造空间，但是应该是可以减少文件的多次读取的I/O压力.
    """

    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]

    net = {}
    current = input_image

    # 抽取VGG19的参数->并且使得image矩阵在VGG19中走一遍.
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)
        net[name] = current
    # 保证网络结构的完整性，进行net完整性检查.
    assert len(net) == len(layers)
    return net


def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')


def preprocess(image):
    return image - MEAN_PIXEL


def unprocess(image):
    return image + MEAN_PIXEL
