# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:42:46 2020

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

########################### LEARNING RATE SCHEDULER ###########################

# Function for learning rate decay. The learning rate will reduce by a factor of 0.1 every 10 epochs.
def schedlr(epoch, lr):
    new_lr = 0.001 * (0.1)**(floor(epoch/10))
    return new_lr

############################### HYPERPARAMETERS ###############################

IMG_SIZE = 224
BATCH = 8
EPOCHS = 100

################################### DATASET ###################################

# Paths for relevant datasets to load in
train_dataset = r'dataset/samples_train'
test_dataset = r'dataset/samples_test'
val_dataset = r'dataset/samples_val'

# Make a list of the test folders to be used when predicting the model. This will be fed into the prediction
# flow to generate the stitched image based off the predictions of the patches fed into the network
_, test_fol, _ = next(os.walk(test_dataset))

# Load in the relevant datasets 
X_train, Y_train, X_test, Y_test, X_val, Y_val = DatasetLoad(train_dataset, test_dataset, val_dataset)
        
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
checkpoint_path = os.path.join(dname, 'models', 'resunet.{epoch:02d}-{f1:.2f}.hdf5')
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False)

callbacks =[
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    LearningRateScheduler(schedlr, verbose=1),
    checkpoint]

# Fit the network to the training dataset. The validation dataset can be used instead of a validataion split
model.fit(X_train, Y_train, validation_split=0.1, batch_size=BATCH, epochs=EPOCHS, callbacks=callbacks)

# Uncomment lines 84-85 and comment line 78 to run a previous model for prediction. Uncommenting lines 84-86 will 
# allow for training continuation in the event that the training was interuppted for whatever reason. If this is the 
# case, please comment out line 78 as well

# latest_checkpoint = r'models/resunet.16-0.93.hdf5'
# model = tf.keras.models.load_model(latest_checkpoint)
# model.fit(X_train, Y_train, validation_split=0.1, batch_size=BATCH, epochs=EPOCHS, callbacks=callbacks, initial_epoch=8)

########################### PREDICTION AND RESULTS ############################

# If previous results exist, delete them so the results won't be mixed up
if os.path.isdir(r'results') == True:
    shutil.rmtree('results')

# Make new results directory along with sub directories for each of the test images
if os.path.isdir(r'results') == False:
    os.mkdir('results')

# Generate the predicted masks for each of the test images and save the patches for use when restitching the image
for i in test_fol:    
    if os.path.isdir('results/%s' % i) == False:
        os.mkdir('results/%s' % i)
    
    save_dir = os.path.join('results', str(i))
    
    pred_test = model.predict(X_test[i], verbose=1)
    pred_test_mask = (pred_test > 0.4).astype(np.uint8)
    
    for n in range(len(pred_test_mask)):
        outputmask = np.squeeze(pred_test_mask[n]*255)
        saveimg = Image.fromarray(outputmask, 'L')
        saveimg.save(os.path.join(save_dir, str(n)).replace('\\','/') + '.png', 'PNG')

# Loop through the entire test prediction dataset and feed the images as an input to the stitching function
for i in test_fol: 
    results_dir = os.path.join('results', str(i))
    imgstitch(results_dir)


