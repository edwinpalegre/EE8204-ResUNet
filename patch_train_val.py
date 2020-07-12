# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 02:30:54 2020

@author: edwin.p.alegre
"""

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

################################## PARAMETERS #################################

IMAGE_SIZE = 224
OVERLAP = 14
NUM_OF_PATCHES = 10
NUM_OF_SAMPLES = 30000

#################################### PATHS ####################################

# Just change these paths to get different samples for training and validation datasets
sample_dataset = r'dataset/samples_train'
train_dataset = r'dataset/training'

# Verify if sample folder exists. 
if os.path.isdir(sample_dataset) == False:
    os.chdir('dataset/')
    os.mkdir('samples_train')
    os.mkdir('samples_train/image')
    os.mkdir('samples_train/mask')
    sampled_image_path = os.path.join(sample_dataset, 'image').replace('\\','/')
    sampled_mask_path = os.path.join(sample_dataset, 'mask').replace('\\','/')
else:
    sampled_image_path = os.path.join(sample_dataset, 'image').replace('\\','/')
    sampled_mask_path = os.path.join(sample_dataset, 'mask').replace('\\','/')

# IMAGE FILE PARAMETERS
_, _, train_files = next(os.walk(os.path.join(train_dataset, 'image')))
training_imgs = len(train_files)

def get_randompatch():
    """
    Function that randomly generates a patch of size [224 x 224 x NUM_OF_CHANNELS] for the training image
    and its corresponding mask

    Returns
    -------
    cropped_img : PIL IMAGE
        Cropped image of size [224, 224, 3]
    cropped_mask : PIL IMAGE
        Cropped corresponding mask of size [224, 224, 1]
    mean_img_val : FLOAT
        Mean value of the cropped image. To be used for occlusion check
    id_name : INT
        ID of the image and mask

    """
    # Randomly choose image from list of all available training images
    id_name = np.random.randint(1, training_imgs + 1)
    
    # Set image and corresponding mask paths
    image_path = os.path.join(train_dataset, 'image', str(id_name)).replace('\\','/') + '.png'
    mask_path = os.path.join(train_dataset, 'mask', str(id_name)).replace('\\','/') + '.png'
    
    # Read image and map, get shape to help with size restrictions
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
    
    # Generate randomized crop coordinates
    start = (np.random.rand(1) * (image_shape[0] - IMAGE_SIZE, image_shape[1] - IMAGE_SIZE)).astype('int')
    end = start + (IMAGE_SIZE, IMAGE_SIZE)
    
    # Crop the actual sampled image and mask from the original, save with new id name
    cropped_img = img.crop((start[0], start[1], end[1], end[1]))
    cropped_mask = mask.crop((start[0], start[1], end[1], end[1]))
    mean_img_val = np.mean(cropped_img)
    
    # fig = plt.figure(figsize=(20,10))
    # ax = fig.add_subplot(1, 2, 1)
    # ax.imshow(cropped_img)
    # ax = fig.add_subplot(1, 2, 2)
    # ax.imshow(cropped_mask, cmap='gray')
    # print(mean_img_val)

    return cropped_img, cropped_mask, mean_img_val, id_name

'''
To get around the issue of some occluded images in the dataset which DO have corresponding masks that
dont actually have the roads in the original image due to occlusion, we verify that the image patch is not 
mainly occluded by taking the mean value of the image. To ensure that we don't fully eliminate images with
a bright background naturally, the threshold is set particularily high. This will let SOME patches with occlusion
pass through, but it should be offset by the amount of patches that actually are proper ground truths
'''
for new_id_name in tqdm(range(1, NUM_OF_SAMPLES + 1)):
    threshold_value = 255
    while(threshold_value > 150):
        cropped_img, cropped_mask, threshold_value, id_name = get_randompatch()
    
    '''
    This part of the program is only used to help verify which image patch you are looking at relative to the original 
    image it was sampled from. It will set the name to contain the ID of the original image that it was sampeld from 
    so the user can correlate the two for inspection, if needed
    '''
    # cropped_img.save(os.path.join(sampled_image_path, str(new_id_name) + '-' + str(id_name)) + '.png', 'PNG')
    # cropped_mask.save(os.path.join(sampled_mask_path, str(new_id_name) + '-' + str(id_name)) + '.png', 'PNG')
    
    cropped_img.save(os.path.join(sampled_image_path, str(new_id_name)) + '.png', 'PNG')
    cropped_mask.save(os.path.join(sampled_mask_path, str(new_id_name)) + '.png', 'PNG')
    
    '''
    This part of the program is to help determine the samples that have a particularily high mean value. This way, the user
    can inspect the sample to see if the image has a high mean value due to occlusion or if it's naturally occuring
    '''
    # if threshold_value > 100:
    #     print(new_id_name,'-', id_name, ' = ', threshold_value, '*****')
    # else:
    #     print(new_id_name,'-', id_name, ' = ', threshold_value)
    
    new_id_name += 1    

print('Done!')