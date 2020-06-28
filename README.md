# EE8204-ResUNet

## Overview
This is Implementation 1 for the course project for Ryerson Univerity's EE8204 - Neural Network course. This current version is the reimplemetnation of the Deep Residual U-Net used for Road Extraction by Zhang et al. (https://arxiv.org/abs/1711.10684). This project is mainly used to familiarize how Deep Neural Networks and their flow are designed and implemented in Tensorflow 2.

![Residual U-Net Model](https://github.com/edwinpalegre/EE8204-ResUNet/blob/master/images/model.PNG)

The dataset used will be Minh's Massachusetts roads dataset which can be found here (https://www.cs.toronto.edu/~vmnih/data/). A Python script was copied from @gist (https://gist.github.com/Diyago/83919fcaa9ca46e4fcaa632009ec2dbd) to assist with downloading the dataset into the working directory.

There will be 3 main components in this repository:
- model_resunet.py: This program defines and verifies the Deep Residual UNet. It verifies the model by providing a summary, letting the user check the size output at each layer to verify that everything makes sense
- patch_sampler.py: This program generates the required 224x224 samples from the original dataset. It's set to generate 30,000 samples, but can be adjusted as needed. I've also integrated a very simple filtering logic to avoid accidentally including samples generated from occluded images in the dataset.
- train.py: This actually trains the ResUNet.

This method utilizes the *Mean Squared Error* as its loss function. It's optimized using *Stochastic Gradient Descent* (SGD). The initial learning rate is set to 0.001 and is reduced by a factor of 0.1 after every 20 epochs. This can be implemented based off Tensorflow's learning rate schedule. The network is expected to converge after 50 epochs. Furthermore, the authors mention training this network using mini-batches of 8. Based off the fact that a Titan 1080 GPU was used in the original implementation (and since I am using a 2070 Super GPU), it makes sense to adopt the gradient accumulation methodology to train this network using the aforementioned mini-batch size. This has yet to be implemented.

## Program Flow

### The Deep Residual U-Net 
First and foremost, the Res-UNet model needs to be generated. The program **model_resunet.py** generates this model. The full model can be imported from this program and then called using the ResUNet((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)) method. It should be noted that tf.keras.layers.Lambda() was used to convert the uint8 datatype of the read image into float for use in network. In the event that a data generator is used, this conversion can happen internally and thus, this line can be commented out as needed.

Below is the output of the model.summary, along with the model image for reference.

Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 224, 224, 3) 0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 224, 224, 64) 1792        input_1[0][0]                    
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 224, 224, 64) 256         conv2d[0][0]                     
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 224, 224, 64) 256         input_1[0][0]                    
__________________________________________________________________________________________________
activation (Activation)         (None, 224, 224, 64) 0           batch_normalization[0][0]        
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 224, 224, 64) 256         conv2d_2[0][0]                   
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 224, 224, 64) 36928       activation[0][0]                 
__________________________________________________________________________________________________
add (Add)                       (None, 224, 224, 64) 0           batch_normalization_1[0][0]      
                                                                 conv2d_1[0][0]                   
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 224, 224, 64) 256         add[0][0]                        
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 224, 224, 64) 0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 112, 112, 128 73856       activation_1[0][0]               
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 112, 112, 128 512         conv2d_3[0][0]                   
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 112, 112, 128 8320        add[0][0]                        
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 112, 112, 128 0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 112, 112, 128 512         conv2d_5[0][0]                   
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 112, 112, 128 147584      activation_2[0][0]               
__________________________________________________________________________________________________
add_1 (Add)                     (None, 112, 112, 128 0           batch_normalization_4[0][0]      
                                                                 conv2d_4[0][0]                   
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 112, 112, 128 512         add_1[0][0]                      
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 112, 112, 128 0           batch_normalization_5[0][0]      
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 56, 56, 256)  295168      activation_3[0][0]               
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 56, 56, 256)  1024        conv2d_6[0][0]                   
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 56, 56, 256)  33024       add_1[0][0]                      
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 56, 56, 256)  0           batch_normalization_6[0][0]      
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 56, 56, 256)  1024        conv2d_8[0][0]                   
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 56, 56, 256)  590080      activation_4[0][0]               
__________________________________________________________________________________________________
add_2 (Add)                     (None, 56, 56, 256)  0           batch_normalization_7[0][0]      
                                                                 conv2d_7[0][0]                   
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 56, 56, 256)  1024        add_2[0][0]                      
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 56, 56, 256)  0           batch_normalization_8[0][0]      
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 28, 28, 512)  1180160     activation_5[0][0]               
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 28, 28, 512)  2048        conv2d_9[0][0]                   
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 28, 28, 512)  131584      add_2[0][0]                      
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 28, 28, 512)  0           batch_normalization_9[0][0]      
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 28, 28, 512)  2048        conv2d_11[0][0]                  
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 28, 28, 512)  2359808     activation_6[0][0]               
__________________________________________________________________________________________________
add_3 (Add)                     (None, 28, 28, 512)  0           batch_normalization_10[0][0]     
                                                                 conv2d_10[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose (Conv2DTranspo (None, 56, 56, 256)  524544      add_3[0][0]                      
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 56, 56, 512)  0           conv2d_transpose[0][0]           
                                                                 add_2[0][0]                      
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 56, 56, 512)  2048        concatenate[0][0]                
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 56, 56, 512)  0           batch_normalization_11[0][0]     
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 56, 56, 256)  1179904     activation_7[0][0]               
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 56, 56, 256)  1024        conv2d_12[0][0]                  
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 56, 56, 256)  131328      concatenate[0][0]                
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 56, 56, 256)  0           batch_normalization_12[0][0]     
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 56, 56, 256)  1024        conv2d_14[0][0]                  
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 56, 56, 256)  590080      activation_8[0][0]               
__________________________________________________________________________________________________
add_4 (Add)                     (None, 56, 56, 256)  0           batch_normalization_13[0][0]     
                                                                 conv2d_13[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_1 (Conv2DTrans (None, 112, 112, 128 131200      add_4[0][0]                      
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 112, 112, 256 0           conv2d_transpose_1[0][0]         
                                                                 add_1[0][0]                      
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 112, 112, 256 1024        concatenate_1[0][0]              
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 112, 112, 256 0           batch_normalization_14[0][0]     
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 112, 112, 128 295040      activation_9[0][0]               
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 112, 112, 128 512         conv2d_15[0][0]                  
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 112, 112, 128 32896       concatenate_1[0][0]              
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 112, 112, 128 0           batch_normalization_15[0][0]     
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 112, 112, 128 512         conv2d_17[0][0]                  
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 112, 112, 128 147584      activation_10[0][0]              
__________________________________________________________________________________________________
add_5 (Add)                     (None, 112, 112, 128 0           batch_normalization_16[0][0]     
                                                                 conv2d_16[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_2 (Conv2DTrans (None, 224, 224, 64) 32832       add_5[0][0]                      
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 224, 224, 128 0           conv2d_transpose_2[0][0]         
                                                                 add[0][0]                        
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 224, 224, 128 512         concatenate_2[0][0]              
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 224, 224, 128 0           batch_normalization_17[0][0]     
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 224, 224, 64) 73792       activation_11[0][0]              
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 224, 224, 64) 256         conv2d_18[0][0]                  
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 224, 224, 64) 8256        concatenate_2[0][0]              
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 224, 224, 64) 0           batch_normalization_18[0][0]     
__________________________________________________________________________________________________
batch_normalization_19 (BatchNo (None, 224, 224, 64) 256         conv2d_20[0][0]                  
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 224, 224, 64) 36928       activation_12[0][0]              
__________________________________________________________________________________________________
add_6 (Add)                     (None, 224, 224, 64) 0           batch_normalization_19[0][0]     
                                                                 conv2d_19[0][0]                  
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 224, 224, 1)  65          add_6[0][0]                      
==================================================================================================
Total params: 8,059,649
Trainable params: 8,051,329
Non-trainable params: 8,320
__________________________________________________________________________________________________

![ResUnet Model]

### Sampled Training Dataset

### Training Procedure

## Results
