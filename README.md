# EE8204-ResUNet

This is Implementation 1 for the course project for Ryerson Univerity's EE8204 - Neural Network course. This current version is the reimplemetnation of the Deep Residual U-Net used for Road Extraction by Zhang et al. (https://arxiv.org/abs/1711.10684). This project is mainly used to familiarize how Deep Neural Networks and their flow are designed and implemented in Tensorflow 2.

The dataset used will be Minh's Massachusetts roads dataset which can be found here (https://www.cs.toronto.edu/~vmnih/data/). A Python script was copied from https://gist.github.com/Diyago/83919fcaa9ca46e4fcaa632009ec2dbd to assist with downloading the dataset into the working directory.

There will be 3 main components in this repository:
- model_resunet.py: This program defines and verifies the Deep Residual UNet. It verifies the model by providing a summary, letting the user check the size output at each layer to verify that everything makes sense
- utils.py: This includes any functions that deal with image manipulation, the data generator, etc
- train.py: This actually trains the ResUNet.
