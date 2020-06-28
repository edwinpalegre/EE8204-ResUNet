# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 02:44:43 2020

@author: edwin.p.alegre
"""

### LIBRARIES ###
import os
import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf

### PATHS ###
os.chdir('C:/Users/edwin.p.alegre/Google Drive/Synced Folders/Academics/Ryerson University - MASc/Courses/EE8204 - Neural Networks/Course Project/project_build_resunet/dataset')
dataset = os.getcwd()
dataset_train = os.path.join(dataset, 'training')
dataset_test = os.path.join(dataset, 'testing')

### LOSS FUNCTION ###
def dice_coeff(y_true, y_pred, smooth=1):
    flatten_layer = tf.keras.layers.Flatten()
    intersection = tf.math.reduce_sum((flatten_layer(y_true) * flatten_layer(y_pred)))
    union = tf.math.reduce_sum(flatten_layer(y_true)) + tf.math.reduce_sum(flatten_layer(y_pred))
    return ((2. * intersection + smooth) / (union + smooth))
    
def dice_coeff_loss(y_true, y_pred):
    return dice_coeff(y_true, y_pred)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8, image_size=224):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
    
    def __load__(self, id_name):
        # Path
        image_path = os.path.join(self.path, 'image', str(id_name)).replace('\\','/') + '.png'
        mask_path = os.path.join(self.path, 'mask', str(id_name)).replace('\\','/') + '.png'
        
        # Read Image
        image = cv2.imread(image_path)
        if (image.shape[:-1] != (self.image_size, self.image_size)):
            image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Read Mask (as a greyscale)
        mask = cv2.imread(mask_path, 0)
        if (mask.shape != (self.image_size, self.image_size)):
            mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask = np.expand_dims(mask, axis=-1)
        
        # Normalize the image and mask
        image = image/255.0
        mask = mask/255.0
        
        return image, mask
    
    def __getitem__(self, index):
        # In the event 
        if (index + 1) * self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index * self.batch_size
            
        files_batch = self.ids[index * self.batch_size : (index + 1) * self.batch_size]
        
        image = []
        mask = []
        
        for id_name in  files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)
        
        image = np.array(image)
        mask = np.array(mask)
        
        return image, mask
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))
    