# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 02:44:43 2020

@author: edwin.p.alegre
"""

### LIBRARIES ###
import os
import cv2
from tensorflow import keras
from tensorflow.keras.backend import *
from xml.dom import minidom as xml

### PATHS ###
dataset = 'C:/Users/edwin.p.alegre/Google Drive/Synced Folders/Academics/Ryerson University - MASc/Courses/EE8204 - Neural Networks/Course Project/project_built/dataset'
dataset_train = os.path.join(dataset, 'training')
dataset_test = os.path.join(dataset, 'testing')

### LOSS FUNCTION ###
def dice_coeff(y_true, y_pred, smooth=1):
    intersection = K.sum((K.flatten(y_true) * K.flattten(y_pred)))
    union = K.sum(K.flatten(y_true)) + K.sum(K.flattten(y_pred))
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
        image_path = os.path.join(self.path, 'image', 'img-') + id_name + '.png'
        
    