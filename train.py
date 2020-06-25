# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 01:24:50 2020

@author: edwin.p.alegre
"""

import os, sys
sys.path.append('C:/Users/edwin.p.alegre/Google Drive/Synced Folders/Academics/Ryerson University - MASc/Courses/EE8204 - Neural Networks/Course Project/project_build_resunet')
print(sys.path)
import utils
import model_resunet
import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

## Seeding 
seed = 2020
random.seed = seed
np.random.seed = seed
tf.seed = seed

# HYPERPARAMETERS

train_dataset = r'C:/Users/edwin.p.alegre/Google Drive/Synced Folders/Academics/Ryerson University - MASc/Courses/EE8204 - Neural Networks/Course Project/project_build_resunet/dataset/training'
test_dataset = r'C:/Users/edwin.p.alegre/Google Drive/Synced Folders/Academics/Ryerson University - MASc/Courses/EE8204 - Neural Networks/Course Project/project_build_resunet/dataset/testing'

_, _, train_files = next(os.walk(os.path.join(train_dataset, 'image')))
_, _, test_files = next(os.walk(os.path.join(test_dataset, 'image')))
training_imgs = len(train_files)
testing_imgs = len(test_files)

train_ids = list(range(1, training_imgs + 1))
test_ids = list(range(1, testing_imgs + 1))

image_size = 224
batch_size = 8
learning_rate = 0.001
learning_rate_factor = 0.1
epoch_factor = 20
overlap = 14
samples = 30000

# VERIFICATION 
'''
gen = utils.DataGenerator(train_ids, train_dataset, batch_size=batch_size, image_size=image_size)
x, y = gen.__getitem__(0)
print(x.shape, y.shape)
r = random.randint(0, len(x)-1)
fig = plt.figure()
fig.subplots_adjust(hspace=1, wspace=1)
ax = fig.add_subplot(1, 2, 1)
ax.imshow(x[r])
ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(y[r]*255, (image_size, image_size)), cmap="gray")
'''

# TRAINING
train_gen = utils.DataGenerator(train_ids, train_dataset, batch_size=batch_size, image_size=image_size)
test_gen = utils.DataGenerator(test_ids, test_dataset, batch_size=batch_size, image_size=image_size)

train_step = len(train_ids)//batch_size
test_step = len(test_ids)//batch_size

epochs = 50

# MODEL
model = model_resunet.ResUNet((image_size, image_size, 3))
adam = keras.optimizers.Adam()
model.compile(optimizer=adam, loss=utils.dice_coeff_loss, metrics=[utils.dice_coeff])
model.summary()

model.fit_generator(train_gen, validation_data=test_gen, steps_per_epoch=train_step, validation_steps=test_step, epochs=epochs)
model.save_weights("ResUNet.h5")

print("\n      Ground Truth            Predicted Value")

for i in range(1, 5, 1):
    ## Dataset for prediction
    x, y = test_gen.__getitem__(i)
    result = model.predict(x)
    result = result > 0.4
    
    for i in range(len(result)):
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(np.reshape(y[i]*255, (image_size, image_size)), cmap="gray")

        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(np.reshape(result[i]*255, (image_size, image_size)), cmap="gray")