# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:33:44 2020

@author: edwin.p.alegre
"""

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Conv2DTranspose, concatenate, Lambda, UpSampling2D, Dropout
from tensorflow.keras import Model, Input
from typing import Tuple
import tensorflow as tf

def dense_conv_layer(feature_map, conv_filter: int):
    bn_1 = BatchNormalization()(feature_map)
    conv_1x1 = Conv2D(filters=conv_filter, kernel_size=(1, 1), strides=(1, 1), padding='same')(bn_1)
    relu_1 = Activation(activation='relu')(conv_1x1)
    