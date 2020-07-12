# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 00:14:37 2020

@author: edwin.p.alegre
"""

################################## LIBRARIES ##################################

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Conv2DTranspose, concatenate, Lambda, UpSampling2D
from tensorflow.keras import Model, Input
from contextlib import redirect_stdout
import tensorflow as tf


############################# CONVOLUTIONAL BLOCK #############################

def conv_block(feature_map):
    
    # Main Path
    conv_1 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(feature_map)
    bn = BatchNormalization()(conv_1)
    relu = Activation(activation='relu')(bn)
    conv_2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(relu)
    
    res_conn = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding='same')(feature_map)
    res_conn = BatchNormalization()(res_conn)
    addition = Add()([res_conn, conv_2])
    
    return addition

############################### RESIDUAL BLOCK ################################

def res_block(feature_map, conv_filter, stride):
    
    bn_1 = BatchNormalization()(feature_map)
    relu_1 = Activation(activation='relu')(bn_1)
    conv_1 = Conv2D(conv_filter, kernel_size=(3,3), strides=stride[0], padding='same')(relu_1)
    bn_2 = BatchNormalization()(conv_1)
    relu_2 = Activation(activation='relu')(bn_2)
    conv_2 = Conv2D(conv_filter, kernel_size=(3,3), strides=stride[1], padding='same')(relu_2)
    

    res_conn = Conv2D(conv_filter, kernel_size=(1,1), strides=stride[0], padding='same')(feature_map)
    res_conn = BatchNormalization()(res_conn)
    addition = Add()([res_conn, conv_2])
    
    return addition

################################### ENCODER ###################################

def encoder(feature_map):
    
    # Initialize the to_decoder connection
    to_decoder = []
    
    # Block 1 - Convolution Block
    path = conv_block(feature_map)
    to_decoder.append(path)
    
    # Block 2 - Residual Block 1
    path = res_block(path, 128, [(2, 2), (1, 1)])
    to_decoder.append(path)
    
    # Block 3 - Residual Block 2
    path = res_block(path, 256, [(2, 2), (1, 1)])
    to_decoder.append(path)
    
    return to_decoder

################################### DECODER ###################################

def decoder(feature_map, from_encoder):
    
    # Block 1: Up-sample, Concatenation + Residual Block 1
    main_path = UpSampling2D(size=(2,2), interpolation='bilinear')(feature_map)
    # main_path = Conv2DTranspose(filters=256, kernel_size=(2,2), strides=(2,2), padding='same')(feature_map)
    main_path = concatenate([main_path, from_encoder[2]], axis=3)
    main_path = res_block(main_path, 256, [(1, 1), (1, 1)])
    
    # Block 2: Up-sample, Concatenation + Residual Block 2
    main_path = UpSampling2D(size=(2,2), interpolation='bilinear')(main_path)
    # main_path = Conv2DTranspose(filters=128, kernel_size=(2,2), strides=(2,2), padding='same')(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=3)
    main_path = res_block(main_path, 128, [(1, 1), (1, 1)])
    
    # Block 3: Up-sample, Concatenation + Residual Block 3
    main_path = UpSampling2D(size=(2,2), interpolation='bilinear')(main_path)
    # main_path = Conv2DTranspose(filters=64, kernel_size=(2,2), strides=(2,2), padding='same')(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=3)
    main_path = res_block(main_path, 64, [(1, 1), (1, 1)])
    
    return main_path

################################ RESIDUAL UNET ################################

def ResUNet(inputshape):
    
    # Input
    model_input = Input(shape=inputshape)
    model_input_float = Lambda(lambda x: x / 255)(model_input)
    
    # Encoder Path
    model_encoder = encoder(model_input_float)
    
    # Bottleneck
    model_bottleneck = res_block(model_encoder[2], 512, [(2, 2), (1, 1)])
    
    # Decoder Path
    model_decoder = decoder(model_bottleneck, model_encoder)
    
    # Output
    model_output = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding='same')(model_decoder)
    
    return Model(model_input, model_output)

################################ SANITY CHECK #################################

# The last couple of lines are only used for a sanity check. It outputs a summary of the layers to verify
# if each layer is outputting the correct dimension as expected. The final line outputs a graphic of 
# the actual network for a visual reference
    
# model = ResUNet((224, 224, 3))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()
# tf.keras.utils.plot_model(model, to_file='model.png', show_layer_names=True, show_shapes=True, rankdir='TB')