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

The model structure can be verified in the modelsummary.txt, along with the model image for reference shown below.

![ResUnet Model]

### Sampled Training Dataset

### Training Procedure

## Results
