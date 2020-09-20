# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:42:46 2020

@author: edwin.p.alegre
"""

from glob import glob

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import model_resunet
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
        
from math import floor
from tqdm import tqdm
from skimage.io import imread
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import scipy.misc
from PIL import Image
import shutil
from utils import imgstitch, DatasetLoad, load_normalize, parse_image

########################### LEARNING RATE SCHEDULER ###########################

# Function for learning rate decay. The learning rate will reduce by a factor of 0.1 every 10 epochs.
def schedlr(epoch, lr):
    new_lr = 0.001 * (0.1)**(floor(epoch/10))
    return new_lr

############################### HYPERPARAMETERS ###############################

IMG_SIZE = 512
BATCH = 8
EPOCHS = 50
SPLIT = 0.2
AUTOTUNE = tf.data.experimental.AUTOTUNE

################################### DATASET ###################################

# Paths for relevant datasets to load in
train_dataset = r'dataset/samples_train_512'
train_dataset_image = r'dataset/samples_train_512/image/'
train_dataset_mask = r'dataset/samples_train_512/mask/'
val_dataset = r'dataset/samples_val_512'
test_dataset = r'dataset/samples_test_512'

# Make a list of the test folders to be used when predicting the model. This will be fed into the prediction
# flow to generate the stitched image based off the predictions of the patches fed into the network
_, test_fol, _ = next(os.walk(test_dataset))

# Load in the relevant datasets 
# X_train, Y_train, X_test, Y_test, X_val, Y_val = DatasetLoad(train_dataset, test_dataset, val_dataset)

# X_train = X_train[0:3000]
# Y_train = Y_train[0:3000]

train_img = ImageDataGenerator(rescale=1.0/255.0, validation_split=SPLIT)
train_mask = ImageDataGenerator(validation_split=SPLIT)



img_gen = train_img.flow_from_directory(train_dataset_image, 
                              target_size=(IMG_SIZE, IMG_SIZE), 
                              color_mode='rgb',
                              batch_size=4,
                              class_mode=None)

mask_gen = train_mask.flow_from_directory(train_dataset_mask, 
                              target_size=(IMG_SIZE, IMG_SIZE), 
                              color_mode='grayscale',
                              batch_size=4,
                              class_mode=None)

train = zip(img_gen, mask_gen)

################################ RESIDUAL UNET ################################

sgd_optimizer = Adam()

# Metrics to be used when evaluating the network
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
f1 = F1Score(num_classes=2, name='f1', average='micro', threshold=0.4)

# Instantiate the network 
model = model_resunet.ResUNet((IMG_SIZE, IMG_SIZE, 3))
model.compile(optimizer=sgd_optimizer, loss='binary_crossentropy', metrics=['accuracy', precision, recall, f1])
model.summary()

# Callacks to be used in the network. Checkpoint can be adjusted to save the best (lowest loss) if desired. 
checkpoint_path = os.path.join(dname, 'models', 'resunet_512.{epoch:02d}-{f1:.2f}.hdf5')
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True)

callbacks =[
    # tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    # tf.keras.callbacks.TensorBoard(log_dir='logs'),
    LearningRateScheduler(schedlr, verbose=1),
    checkpoint]

# # Fit the network to the training dataset. The validation dataset can be used instead of a validataion split
model.fit(train, epochs=EPOCHS, callbacks=callbacks, steps_per_epoch=2000)
