# EE8204-ResUNet

## Overview
This is Implementation 1 for the course project for Ryerson Univerity's EE8204 - Neural Network course. This current version is the reimplemetnation of the Deep Residual U-Net used for Road Extraction by Zhang et al. (https://arxiv.org/abs/1711.10684). This project is mainly used to familiarize how Deep Neural Networks and their flow are designed and implemented in Tensorflow 2.

![Residual U-Net Model](https://github.com/edwinpalegre/EE8204-ResUNet/blob/master/images/model.PNG)

The dataset used will be Minh's Massachusetts roads dataset which can be found here (https://www.cs.toronto.edu/~vmnih/data/). A Python script was copied from @gist (https://gist.github.com/Diyago/83919fcaa9ca46e4fcaa632009ec2dbd) to assist with downloading the dataset into the working directory.

There will be 6 main components in this repository:
- model_resunet.py: This program defines and verifies the Deep Residual UNet. It verifies the model by providing a summary, letting the user check the size output at each layer to verify that everything makes sense
- train.py: This actually trains the ResUNet and sets all the relevant parameters. It also generates the stitched result after training
- dataset_downloader.py: Downloads the dataset from Mnih's UofT website
- patch_train_val.py: This program generates the required 224x224 samples from the original dataset. It's set to generate 30,000 samples, but can be adjusted as needed. I've also integrated a very simple filtering logic to avoid accidentally including samples generated from occluded images in the dataset. Running the program as is will generate teh sampled training dataset. If a sampled validation dataset instead of a validation split is wanted instead, change the first couple of lines as mentioned in the instructions below. 
- path_test.py: This program samples the test dataset of size 224x224 with an overlap of 14 pixels. 
- utils.py: Contains functions that compile the dataset for use in the ResUNet to make sure it's compliant with Tensorflow as an input. It also houses the stitching function needed for generating the final result from the sampled test dataset

This method utilizes the *Mean Squared Error* as its loss function. It's optimized using *Stochastic Gradient Descent* (SGD). The initial learning rate is set to 0.001 and is reduced by a factor of 0.1 after every 20 epochs. This can be implemented based off Tensorflow's learning rate schedule. The network is expected to converge after 50 epochs. 

## Program Flow

### Specs and Dependencies
- Tensorflow 2.2.0 
- Python 3.8.3
  - numpy=1.18.5
  - tqdm=4.46.1 (Optional, really useful to just see how long it will take to download and load the datasets as it does take a while)
  - scikit-image=0.17.2
  - matplotlib=3.2.1 (Optional, used mainly for debugging and displaying images. Results and samples are already saved anyways so this was only used in debugging stage)
  - Pillow=7.1.2
  - bs4=0.0.1
  - pydotplus=2.0.2 & graphviz=0.14 (Optional, used for tf.keras.plot_model(). A visual model can be seen when using Tensorboard so this really is super optional. If you do decide to 
  download these utilites, make sure that you also run the MSI installer for graphviz to get the executables. Following this, find the graphviz bin folder and add that to your 
  PATH in your system variables afterwards)
- CUDA (Optional, only install if using local machine with a CUDA capable GPU)
 
 Relevant Hardware
 - OS: Windows 10 OS
 - GPU: NVIDIA 2070 Super GPU
 - CPU: AMD Ryzen 5 3600
 - RAM: 32 GB  

### Step by Step Implementation
1. Clone the repo to your project folder and install all dependencies as mentioned above
2. Create a 'dataset' folder within your project directory
3. Run 'dataset_downloader.py'. This will download the Massachusetts Roads Dataset from the UofT website. In the event that you get a server error, wait about an hour. The website does experience some downtime every once in a while. The program will save everything to your 'dataset' folder, which really makes life easy.
4. Run 'patch_train_val.py'. This program may need to be run twice. The initial run will generate sampled patches of the images of size 224x224 for both the training images and masks. It's been set to generate 30,000 samples so it does take a while. This can be changed to your desired number of samples. The second run is optional as it is strictly for creating a sampled validation dataset. I don't really recommend this, as you can just use a validation split of your training dataset instead. If you do decide to use the validation dataset, make sure you change lines 30-38 to 'samples_val' instead of 'samples_train' so it doesn't overwrite your sampled training dataset. Furthermore, make sure to change the number of samples as you don't need 30,000 validation samples.
5. Run 'patch_test.py'. This will generate the patches from the test dataset to be compliant with the input size restrictions of the ResUNet.
6. Run 'train.py'. This will load in the sampled datasets, train the network, and provide output results. Training time will vary depending on the hardware used. For a setup similar to mine, the load time for the dataset takes around 15-20 minutes. Each training epoch takes around 20 minutes with a validation check that lasts a minute or two. The network seems to converge at around 10-15 epochs depending on how training goes. I'm currently testing the network with an adjusted learning rate that decays every 10 epochs, which may extend the training time 
7. After running 'train.py' a results directory will be generated automatically. There will be 13 folders housing the predicted mask patches as well as the stitched output. The final result stitched output image will be located in the same directory as the sampled masks with the name 'output.png'

## Problem Statement
Road extraction can be plainly summarized as binary semantic segmentation, where the road would be classified as the foreground and everything else as the background. It's important to be able to do this on the fly as many applications may need up-to-date road maps of certain areas to coordinate rescue efforts after natural disasters, humanitarian aid, changing infrastructure, and urban planning. As such, we seek to create a mask of the roads using a neural network by using remote sensing images as an input. Remote sensing images can vary on source and channels, but this implementation focuses on satellite imagery. Satellite imagery can capture very high resolution RGB images, hyperspectral images, and multispectral images just to list a few. There are advantages to each type of image, but VHR RGB images are utilized as it is simple to handle the data for a 3 channel image. Discussed further in the Future Works section, multispectral images may be utilized in future versions of this project for their ability to capture information that are occluded by vegetation or other obstacles in RGB images. The following image showcases the RGB satellite image, followed by the ground truth mask, and then the results from 3 different methods. 

![Results](https://github.com/edwinpalegre/EE8204-ResUNet/blob/master/images/results.PNG)

Since road extraction is a segmentation task, many approaches utilize the U-Net as the network structure. The U-Net is renowned for its segmentation capabilities, which is made possible by its expansion and contraction paths (also refered to as the encoder and decoder path) which allow for both the contextualization and localization of the image. However, as with many deep neural networks, a signifantly deep network is bound to suffer from the vanishing gradient issue, where the gradient minimizes to 0 at a certain point. This halts the updates to the weights as the gradient value effectively zeros out the backpropagation update increments. 

The two mainly utilized methods of solving this issue come in the form of the residual connection and dense connections. A residual connection links the input of a layer with its output through a skip connection. This skip connection allows for the propagaton of an identity mapping from the input to the output, thereby providing local features to help mitigate the vanishing gradient problem. In the case that the dimensionality changes, linear operations must be applied in order to account for this dimension change. Any operation done in the skip connection must adhere to a stride of 2 with a 1x1 kernel. The dense connection can be considered a more advanced residual connection, where every layer is connected to each subsequent layer after it. Thus, the network is densely connected. The functionality is similar to that of the residual connection.

by combining the structure of the U-Net with the block representations of the residual connection, the advantages of both concepts can be utilized to approach this problem. Hence

## The Approach

### The Deep Residual U-Net 
First and foremost, the Res-UNet model needs to be generated. The program **model_resunet.py** generates this model. The full model can be imported from this program and then called using the ResUNet((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)) method. It should be noted that tf.keras.layers.Lambda() was used to convert the uint8 datatype of the read image into float for use in network. In the event that a data generator is used, this conversion can happen internally and thus, this line can be commented out as needed.

The model structure can be verified in the modelsummary.txt, along with the model image for reference shown below.

![ResUnet Model](https://github.com/edwinpalegre/EE8204-ResUNet/blob/master/images/resunet.png)

### Sampled Training Dataset
The original paper states that from Mnih's dataset, a sampled training dataset of 30,000 images were to be generated. There are occlusions in approximately 100 of these images. This poses as issue since their corresponding masks classify areas under these occlusions as having roads. Thus, it is in our interest to minimize the amount of occluded samples in our training dataset. To accomodate this, a simple filtering operation was created. After the 224x224 image patch sample is created, the mean of all the pixels in the image is taken. The reasoning is that these areas are primarily dark, thus their mean should be fairly low. The occlusions are large noticeable patches of white pixels, bringing up the mean by a considerable amount. After some testing, it was found that there were some original images that were representative of different regions that had brigetr pixels (i.e fields of yellow wheat). After some testing, a threshold of 150 was chosen. While it will not fully eliminate these occluded images, it will minimize the probability that a majorly occluded image will be included in the dataset. It also ensures that actual images that are naturally bright do not get fully filtered out.

### Training Procedure
As stated previously, 30,000 224x224 RGB sampled images were used for training. Thus, the model takes a 224x224x3 input image and returns a binary segmented map of the same size. The network utilizes the mean squared error as its loss function and optimizes it using the stochastic gradient decent algorithm. It was trained with a learning rate of 0.001 and reduced by a factor of 0.1 after each 20 epochs. Convergence is expected to occcur after 50 epochs. 

This approach was later changed as the results were not adequate. This is discussed in the results section. Instead, the Adam optimizer was used along with the binary cross entropy loss function.

The images are fed into the network as a single variable, where the RGB images are a [30000x224x224x3] array, and its correspnsding mask is treated the same albeit with a single channel instead of 3. It is possible to use Tensorflow's batch loading API, but there is no difference. These are passed into the model.fit() method for training. Callbacks used include the ModelCheckpoint(), which saves the best model based off the validation loss, EarlyStopping(), which monitors the validation loss with a patience of 5, and TensorBoard(), which provides a visualization of how training is progressing. The learning rate schedule is also included in the callbacks but is defined as a function at the beginning of train.py.

## Results
Achieving results that were equivalent to the original paper were not successful. Using the methods outlined in their paper, that is to say, utilizing Stochastic Gradient Descent as the optimizer and minimizing the Mean Squared Error loss function did not result in the expected results outlined in the paper. A validation F1 score of 0.3 was achieved. The validation precision hovered at around 0.4 - 0.5 while the recall was significantly lower at 0.15 - 0.2. The paper did use a relaxed version of these metrics, where a range of 3 pixels was allowed to be misclassified but still count towards the correct road label. Based on the low score, it would not be probable that my implementation would have even approached the same results that Zhang et al did. Furthermore, they utilized a breakeven point between the precision and recall scores. This is because the F1 score is a function of these two metrics. Thus, F1 is maximized and achieves a decent tradeoff if the two metrics are equal. Zhang et al achieved a breakeven point of 0.9187, that is, achieving a precision, recall, and F1 score of 0.9187. 

This could be attributed to a couple of issues. While I did use the same training method they did (same number of samples, same dataset, same method of sampling training images, same batch size, same learning rate and schedule, etc) I used the Tensorflow implementation of MSE. Immediately I noticed how small my training loss (error) was, ending the first epoch with a loss of 0.01 I believe. The authors may have calculated the MSE differently than Tensorflow did, but if so, they never specified. Furthermore, there were some parts of the model where I required clarification. 

First and foremost, they mention that this model was based off the U-net as it's structure utilizes an encoding and decoding path. The U-Net requires an up-sampling operation of some kind. There were two main approahces to this. My initial iteration involved the use of the Conv2dTranspose layer (which applied a transposed convolution to upsample the input image to the correct dimensions needed for the output). The main issue with this method is that the COnv2DTranspose, from what I could figure out, required training to properly learn how to convolutionally transpose the input image properly. This may have attributed to my less than stellar metrics. Instead, I opted for the UpSampling2D layer which uses interpolation methods to upsample the input image. I settled on bilinear interpolation over nearest neighbours as it provided better results. 

The second disrepancy comes in the form of the residual connections utilized at each block. In the paper, they never specify the identity mapping they used for their residual connection. They mention is that a common identity mapping used is h_l = x and that they use a full preactivation residual unit which uses this same mapping. That is, the identity mapping is the input feature map. This doesn't make sense in the case of this network. The identity mapping is added to the output layer to provide context when the neural network grows significantly deep. This ensures that the gradient degradation is mitigated. However, since the dimensionality of the input and output are different, this type of indentity mapping cannot be utilized. Some form of linear operation must be applied to abide by the required dimesionality needed for the two feature maps to be summed. This is mentioned in the original paper "Deep Residual Learning for Image Recognition". Thus, after some research, I settled on applying a Conv2D layer with a stride of 2 with a 1x1 kernel followed by BatchNormalization. Again, this is mentioned in the original paper as one of its caveats. It is unknown if Zhang et al employed some other form of linear operations to abide by dimentionality which could affect the results.

Since my original approach didn't work too well, I decided to pivot. Instead of employing MSE as my loss function, I decided to use one of semantic semgmentation's most widely used loss functions, the Cross Entropy loss. Since this version of road extraction is binary (with road being classified as '1' and everything else as '0'), I used the Binary Cross Entropy loss function. Furthermore, I decided to use the Adam optimizer as well. Immediately, my results were more impressive. Using the same training sampled training data with a 10% validation split and the aforementioned changes, my initial epoch achieved a training F1 score of roughly 0.6! The network seems to converge after 10-15 epochs (I've yet to change the learning rate schedule so the learning rate doesn't decay at all during training, as of now). The original paper stated that their network converged after 50 epochs of training. After training, the training F1 score achieved is around 0.91 with the precision and recall being nearly the same, give or take a percent or two. The validation F1 score is a bit lower at 0.85 - 0.88, depending on the run. However, this is a huge improvement from my initial implemetation. It should be noted that I opted to use a validation split instead of the sampled validation dataset as it provided better results. Using the validation dataset tends to end training early as my program is set to monitor the validation loss and end training if it doesn't improve within 5 epochs. If the validation set is used, the network tends to converge at around 5-8 epochs with a training F1 score of 0.7 - 0.8. Thus, it's recommended that a validation split be used. The best model is provided in this repository and can be loaded in using the train.py program. Previous models were also uploaded for reference. The results from the best run can be seen below. 

There are still issues to be fixed. First of all, a second glance should be given at the original method as it should be reproducable. I'm unsure as to why I haven't achieved the same results as the authors did. The results also suffer from the traditional road extraction issues. Mainly, the fact that the network cannot classify roads with occlusions such as trees or shadows. There also seems to be an issue with classifying parking lots and spaces. This could be partially attributed to the dataset as some of the images classify commercial and residential parking spaces as roads while other images do not. There is alos noise that is produced, although this may be alleviated through post processing techniques such as erosion & dilation. however, current efforts have not provided the best results when this was attempted.

Future work may include the use of conditional random fields (CRFs) for post processing. Furthermore, to help alleviate the occlusions, multispecral satellite images may be useable as these types of images do contain information past the visible spectrum. There are also different architectures that should be implemented like the GL-Dense-UNet, which boasts more robust results than this paper. 

### Model Predicitions
Ground Truth
![Ground Truth](https://github.com/edwinpalegre/EE8204-ResUNet/blob/master/images/groundtruths/7.png)

Predicted Mask
![Mask](https://github.com/edwinpalegre/EE8204-ResUNet/blob/master/images/finalresults/7.png)
