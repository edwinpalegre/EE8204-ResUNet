# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 12:29:43 2020

@author: edwin.p.alegre
"""

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import model_resunet
import numpy as np
import tensorflow as tf
from math import floor
from tqdm import tqdm
from skimage.io import imread
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow_addons.metrics import F1Score
import random
import scipy.misc
from PIL import Image
import shutil
from utils import imgstitch, DatasetLoad

# Paths for relevant datasets to load in
train_dataset = r'dataset/samples_train_512'
test_dataset = r'dataset/samples_test_512'
val_dataset = 0

# Make a list of the test folders to be used when predicting the model. This will be fed into the prediction
# flow to generate the stitched image based off the predictions of the patches fed into the network
_, test_fol, _ = next(os.walk(test_dataset))

# Load in the relevant datasets 
X_train, Y_train, X_test, Y_test, X_val, Y_val = DatasetLoad(train_dataset, test_dataset, val_dataset)

