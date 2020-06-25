# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 02:30:54 2020

@author: edwin.p.alegre
"""

import os, sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image


IMAGE_SIZE = 224
OVERLAP = 14
NUM_OF_PATCHES = 10

sample_dataset = r'C:/Users/edwin.p.alegre/Google Drive/Synced Folders/Academics/Ryerson University - MASc/Courses/EE8204 - Neural Networks/Course Project/project_build_resunet/dataset/samples'
train_dataset = r'C:/Users/edwin.p.alegre/Google Drive/Synced Folders/Academics/Ryerson University - MASc/Courses/EE8204 - Neural Networks/Course Project/project_build_resunet/dataset/training'

if os.path.isdir(sample_dataset) == False:
    os.chdir('C:/Users/edwin.p.alegre/Google Drive/Synced Folders/Academics/Ryerson University - MASc/Courses/EE8204 - Neural Networks/Course Project/project_build_resunet/dataset/')
    os.mkdir('samples')
    os.mkdir('samples/image')
    os.mkdir('samples/mask')
    sampled_image_path = os.path.join(sample_dataset, 'image').replace('\\','/')
    sampled_mask_path = os.path.join(sample_dataset, 'mask').replace('\\','/')
else:
    sampled_image_path = os.path.join(sample_dataset, 'image').replace('\\','/')
    sampled_mask_path = os.path.join(sample_dataset, 'mask').replace('\\','/')
    
_, _, train_files = next(os.walk(os.path.join(train_dataset, 'image')))
training_imgs = len(train_files)   
new_id_name = 1 

#### START LOOP HERE ####

id_name = np.random.randint(1, training_imgs + 1)

image_path = os.path.join(train_dataset, 'image', str(id_name)).replace('\\','/') + '.png'
mask_path = os.path.join(train_dataset, 'mask', str(id_name)).replace('\\','/') + '.png'

img = Image.open(image_path)
mask = Image.open(mask_path)

image_shape = img.size
mask_shape = mask.size


# Display Full Image and Mask
# fig = plt.figure(figsize=(20,10))
# ax = fig.add_subplot(1, 2, 1)
# ax.imshow(img)
# ax = fig.add_subplot(1, 2, 2)
# ax.imshow(mask, cmap='gray')

# Restrict range of possible values
max_xdim = image_shape[0] - IMAGE_SIZE
max_ydim = image_shape[1] - IMAGE_SIZE

patches = []

#for patches in range(NUM_OF_PATCHES):
start = (np.random.rand(1) * (image_shape[0] - IMAGE_SIZE, image_shape[1] - IMAGE_SIZE)).astype('int')
end = start + (IMAGE_SIZE, IMAGE_SIZE)
cropped_img = img.crop((start[0], start[1], end[1], end[1]))
cropped_mask = mask.crop((start[0], start[1], end[1], end[1]))
cropped_img.save(os.path.join(sampled_image_path, str(id_name)) + '.png', 'PNG')
cropped_mask.save(os.path.join(sampled_mask_path, str(id_name)) + '.png', 'PNG')


fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(1, 2, 1)
ax.imshow(cropped_img)
ax = fig.add_subplot(1, 2, 2)
ax.imshow(cropped_mask, cmap='gray')