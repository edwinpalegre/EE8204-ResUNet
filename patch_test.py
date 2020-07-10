# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 20:01:21 2020

@author: edwin.p.alegre
"""


import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

### PARAMETERS ###
IMG_SIZE = 224
OVERLAP = 14

test_dataset = r'dataset/testing'
test_patch_dataset = r'dataset/test_patch'


if os.path.isdir(test_patch_dataset) == False:
    os.chdir('dataset/')
    os.mkdir('test_patch')

    
# IMAGE FILE PARAMETERS
_, _, test_files = next(os.walk(os.path.join(test_dataset, 'image')))
test_imgs = len(test_files)
test_ids = list(range(1, test_imgs + 1))

for id_name in tqdm(range(1, len(test_ids) + 1)):

    row_id_name = 1
    col_id_name = 1
    
    image_path = os.path.join(test_dataset, 'image', str(id_name)).replace('\\','/') + '.png'
    mask_path = os.path.join(test_dataset, 'mask', str(id_name)).replace('\\','/') + '.png'
    
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
    
    x1 = 0
    y1 = 0
    x2 = IMG_SIZE
    y2 = IMG_SIZE
    
    if os.path.isdir(r'dataset/test_patch/' + str(id_name)) == False:
            os.mkdir('dataset/test_patch/' + str(id_name))
            os.mkdir('dataset/test_patch/' + str(id_name) + '/image')
            os.mkdir('dataset/test_patch/' + str(id_name) + '/mask')
    
    while y2 < image_shape[0]:
        while x2 < image_shape[0]:
            
            test_image_path = os.path.join('dataset/test_patch/', str(id_name), 'image').replace('\\','/')
            test_mask_path = os.path.join('dataset/test_patch/', str(id_name), 'mask').replace('\\','/')
            
            # Crop the actual sampled image and mask from the original, save with new id name
            cropped_img = img.crop((x1, y1, x2, y2))
            cropped_mask = mask.crop((x1, y1, x2, y2))
            
            # fig = plt.figure(figsize=(20,10))
            # ax = fig.add_subplot(1, 2, 1)
            # ax.imshow(cropped_img)
            # ax = fig.add_subplot(1, 2, 2)
            # ax.imshow(cropped_mask, cmap='gray')
            
            cropped_img.save(os.path.join(test_image_path, str(row_id_name) + str(col_id_name)) + '.png', 'PNG')
            cropped_mask.save(os.path.join(test_mask_path, str(row_id_name) + str(col_id_name)) + '.png', 'PNG')
            
            col_id_name += 1
            x1 += IMG_SIZE - OVERLAP
            x2 += IMG_SIZE - OVERLAP
        
        x1 = 0
        x2 = IMG_SIZE
        
        col_id_name = 1
        row_id_name += 1
        
        y1 += IMG_SIZE - OVERLAP
        y2 += IMG_SIZE - OVERLAP
    