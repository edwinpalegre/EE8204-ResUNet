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


def schedlr(epoch, lr):
    new_lr = 0.001 * (0.1)**(floor(epoch/20))
    return new_lr

# HYPERPARAMETERS
IMG_SIZE = 224
BATCH = 8
OVERLAP = 14
EPOCHS = 100

# DATASET CREATION
train_dataset = r'dataset/samples'
test_dataset = r'dataset/test_patch'
_, test_fol, _ = next(os.walk(test_dataset))

X_train, Y_train, X_test, Y_test = DatasetLoad(train_dataset, test_dataset)
        
# RESIDUAL U-NET 
sgd_optimizer = Adam()

precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
f1 = F1Score(num_classes=2, name='f1', average='micro', threshold=0.4)

model = model_resunet.ResUNet((IMG_SIZE, IMG_SIZE, 3))
model.compile(optimizer=sgd_optimizer, loss='binary_crossentropy', metrics=['accuracy', precision, recall, f1])
model.summary()

checkpoint_path = os.path.join(dname, 'models', 'resunet.{epoch:02d}-{val_loss:.2f}.hdf5')
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True)

callbacks =[
    tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    LearningRateScheduler(schedlr, verbose=1),
    checkpoint]

model.fit(X_train, Y_train, validation_split=0.1, batch_size=BATCH, epochs=EPOCHS, callbacks=callbacks)

# PREDICTION AND RESULTS

# latest_checkpoint = r'models/resunet.61-0.10.hdf5'
# model = tf.keras.models.load_model(latest_checkpoint)



if os.path.isdir(r'results') == True:
    shutil.rmtree('results')

if os.path.isdir(r'results') == False:
    os.mkdir('results')

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

for i in test_fol: 
    results_dir = os.path.join('results', str(i))
    imgstitch(results_dir)

# ix = random.randint(1, len(X_test))
# ix = 1
# fig = plt.figure()
# fig.subplots_adjust(hspace=1, wspace=1)
# ax = fig.add_subplot(2, 2, 1)
# ax.imshow(X_test['1'][0])
# ax = fig.add_subplot(2, 2, 2)
# ax.imshow(np.squeeze(Y_test['1'][0]), cmap="gray")
# ax = fig.add_subplot(2, 2, 3)
# ax.imshow(np.squeeze(pred_test_mask[0]), cmap="gray")
