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

![ResUnet Model](https://github.com/edwinpalegre/EE8204-ResUNet/blob/master/images/resunet.png)

### Sampled Training Dataset
The original paper states that from Mnih's dataset, a sampled training dataset of 30,000 images were to be generated. There are occlusions in approximately 100 of these images. This poses as issue since their corresponding masks classify areas under these occlusions as having roads. Thus, it is in our interest to minimize the amount of occluded samples in our training dataset. To accomodate this, a simple filtering operation was created. After the 224x224 image patch sample is created, the mean of all the pixels in the image is taken. The reasoning is that these areas are primarily dark, thus their mean should be fairly low. The occlusions are large noticeable patches of white pixels, bringing up the mean by a considerable amount. After some testing, it was found that there were some original images that were representative of different regions that had brigetr pixels (i.e fields of yellow wheat). After some testing, a threshold of 150 was chosen. While it will not fully eliminate these occluded images, it will minimize the probability that a majorly occluded image will be included in the dataset. It also ensures that actual images that are naturally bright do not get fully filtered out.

### Training Procedure
As stated previously, 30,000 224x224 RGB sampled images were used for training. Thus, the model takes a 224x224x3 input image and returns a binary segmented map of the same size. The network utilizes the mean squared error as its loss function and optimizes it using the stochastic gradient decent algorithm. It was trained with a learning rate of 0.001 and reduced by a factor of 0.1 after each 20 epochs. Convergence is expected to occcur after 50 epochs. 

A major component of training is the involvement of mini-batches and gradient accumulation. This has yet to be implemented, but it is expected to drastically assist with training our network. The images are fed into the network as a single variable, where the RGB images are a [30000x224x224x3] array, and its correspnsding mask is treated the same albeit with a single channel instead of 3. These are passed into the model.fit() method for training. Callbacks used include the ModelCheckpoint(), which saves the best model based off the validation loss, EarlyStopping(), which monitors the validation loss with a patience of 2, and TensorBoard(), which provides a visualization of how training is progressing. 

## Results
