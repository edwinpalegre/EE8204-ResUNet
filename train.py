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
from tensorflow.keras.optimizers import Adam, SGD, schedules
from tqdm import tqdm
from skimage.io import imread, imshow

# HYPERPARAMETERS
seed = 42
np.random.seed = seed

image_size = 224
batch_size = 8
learning_rate = 0.001
learning_rate_factor = 0.1
epoch_factor = 20
overlap = 14
epochs = 50

train_dataset = r'C:/Users/edwin.p.alegre/Google Drive/Synced Folders/Academics/Ryerson University - MASc/Courses/EE8204 - Neural Networks/Course Project/project_build_resunet/dataset/samples'
test_dataset = r'C:/Users/edwin.p.alegre/Google Drive/Synced Folders/Academics/Ryerson University - MASc/Courses/EE8204 - Neural Networks/Course Project/project_build_resunet/dataset/samples_test'

_, _, train_files = next(os.walk(os.path.join(train_dataset, 'image')))
_, _, test_files = next(os.walk(os.path.join(test_dataset, 'image')))
training_imgs = len(train_files)
test_imgs = len(test_files)
train_ids = list(range(1, training_imgs + 1))
test_ids = list(range(1, test_imgs + 1))

X_train = np.zeros((len(train_ids), image_size, image_size, 3), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), image_size, image_size, 1), dtype=np.bool)

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    X_train[n] = imread(train_dataset + '/image/' + str(id_) + '.png')
    mask = np.zeros((image_size, image_size, 1), dtype=np.bool)
    for mask_file in next(os.walk(train_dataset  + '/mask/')):
        mask_ = imread(train_dataset + '/mask/' + str(id_) + '.png')
        mask_ = np.expand_dims(mask_, axis=-1)
        mask = np.maximum(mask, mask_)
    
    Y_train[n] = mask
    
X_test = np.zeros((len(test_ids), image_size, image_size, 3), dtype=np.uint8)
Y_test = np.zeros((len(test_ids), image_size, image_size, 1), dtype=np.bool)
sizes_test = []
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    X_test[n] = imread(test_dataset + '/image/' + str(id_) + '.png')
    mask = np.zeros((image_size, image_size, 1), dtype=np.bool)
    for mask_file in next(os.walk(test_dataset  + '/mask/')):
        mask_ = imread(test_dataset + '/mask/' + str(id_) + '.png')
        mask_ = np.expand_dims(mask_, axis=-1)
        mask = np.maximum(mask, mask_)
    
    Y_test[n] = mask

########################################

'''
r = random.randint(1, len(train_ids)+1)
fig = plt.figure()
fig.subplots_adjust(hspace=1, wspace=1)
ax = fig.add_subplot(1, 2, 1)
ax.imshow(X_train[r])
ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(Y_train[r]*255, (image_size, image_size)), cmap="gray")
'''

# lr_schedule = schedules.PolynomialDecay(initial_learning_rate=0.001, decay_steps=12000, end_learning_rate=0.00001)
# sgd_optimizer = SGD(learning_rate=lr_schedule)
sgd_optimizer = SGD(learning_rate=0.001)

model = model_resunet.ResUNet((image_size, image_size, 3))
model.compile(optimizer=sgd_optimizer, loss='mean_squared_error', metrics=['accuracy'])
model.summary()

########## BRAND NEW TRAINING ############

checkpoint_path = r'C:/Users/edwin.p.alegre/Google Drive/Synced Folders/Academics/Ryerson University - MASc/Courses/EE8204 - Neural Networks/Course Project/project_build_resunet/models/resunet.{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True)

callbacks =[
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    checkpoint]

########## RESUME TRAINING ############
latest_checkpoint = r'C:/Users/edwin.p.alegre/Google Drive/Synced Folders/Academics/Ryerson University - MASc/Courses/EE8204 - Neural Networks/Course Project/project_build_resunet/models/resunet.06-0.04.hdf5'

model = tf.keras.models.load_model(latest_checkpoint)

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=batch_size, epochs=epochs, initial_epoch=6, callbacks=callbacks)


########################################
idx = random.randint(1, len(X_train))

# pred_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
# pred_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
pred_test = model.predict(X_test, verbose=1)

# pred_train_mask = (pred_train > 0.5).astype(np.uint8)
# pred_val_mask = (pred_val > 0.5).astype(np.uint8)
pred_test_mask = (pred_test > 0.5).astype(np.uint8)

# ix = random.randint(1, len(pred_train_mask))
# fig = plt.figure()
# fig.subplots_adjust(hspace=1, wspace=1)
# ax = fig.add_subplot(2, 2, 1)
# ax.imshow(X_train[ix])
# ax = fig.add_subplot(2, 2, 2)
# ax.imshow(np.squeeze(Y_train[ix]), cmap="gray")
# ax = fig.add_subplot(2, 2, 3)
# ax.imshow(np.squeeze(pred_train_mask[ix]), cmap="gray")


# ix = random.randint(1, len(pred_val_mask))
# fig = plt.figure()
# fig.subplots_adjust(hspace=1, wspace=1)
# ax = fig.add_subplot(2, 2, 1)
# ax.imshow(X_train[int(X_train.shape[0]*0.9):][ix])
# ax = fig.add_subplot(2, 2, 2)
# ax.imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]), cmap="gray")
# ax = fig.add_subplot(2, 2, 3)
# ax.imshow(np.squeeze(pred_val_mask[ix]), cmap="gray")

ix = random.randint(1, len(X_test))
fig = plt.figure()
fig.subplots_adjust(hspace=1, wspace=1)
ax = fig.add_subplot(2, 2, 1)
ax.imshow(X_test[ix])
ax = fig.add_subplot(2, 2, 2)
ax.imshow(np.squeeze(Y_test[ix]), cmap="gray")
ax = fig.add_subplot(2, 2, 3)
ax.imshow(np.squeeze(pred_test_mask[ix]), cmap="gray")

imshow(np.squeeze(pred_test_mask[ix]), cmap="gray")

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

'''

# TRAINING
train_gen = utils.DataGenerator(train_ids, train_dataset, batch_size=batch_size, image_size=image_size)
validation_gen = utils.DataGenerator(validation_ids, train_dataset, batch_size=batch_size, image_size=image_size)

train_step = len(train_ids)//batch_size
validation_step = len(validation_ids)//batch_size

epochs = 50

# MODEL
model = model_resunet.ResUNet((image_size, image_size, 3))
adam = Adam()
model.compile(optimizer=adam, loss=utils.dice_coeff_loss, metrics=[utils.dice_coeff])
model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint('roadseg.h5', verbose=1, save_best_only=True)

callbacks =[
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')]


model.fit_generator(train_gen, validation_data=validation_gen, steps_per_epoch=train_step, validation_steps=validation_step, epochs=epochs)

print("\n      Ground Truth            Predicted Value")

for i in range(1, validation_size + 1, 1):
    ## Dataset for prediction
    x, y = validation_gen.__getitem__(i)
    result = model.predict(x)
    result = result > 0.4
    
    for i in range(len(result)):
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(np.reshape(y[i]*255, (image_size, image_size)), cmap="gray")

        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(np.reshape(result[i]*255, (image_size, image_size)), cmap="gray")
'''
