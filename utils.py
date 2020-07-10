# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 02:44:43 2020

@author: edwin.p.alegre
"""
from PIL import Image
import os
import numpy as np
from skimage.io import imread
from tqdm import tqdm

OGIMG_SIZE = 1500
IMG_SIZE = 224
OVERLAP = 14

def imgstitch(img_path):
    _, _, img_files = next(os.walk(img_path))
    
    img_files = sorted(img_files,key=lambda x: int(os.path.splitext(x)[0]))
    IMG_WIDTH, IMG_HEIGHT = (Image.open(img_path + '/11.png')).size
    
    img = np.zeros((len(img_files), IMG_WIDTH, IMG_HEIGHT), dtype=np.uint8)
    full_img = Image.new('RGB', (1470, 1470))
    x, y = (0, 0)
    
    for n, id_ in enumerate(img_files):
        img = Image.open(img_path + '/' + str(id_))
        if x < 1460:
            full_img.paste(img, (x, y))
            x += IMG_WIDTH - OVERLAP
        if x > 1460:
            x = 0
            y += IMG_WIDTH - OVERLAP
            full_img.paste(img, (x, y))
    
    full_img.save(os.path.join(img_path, 'output') + '.png', 'PNG')
    
def DatasetLoad(train_dataset, test_dataset):
    _, _, train_files = next(os.walk(os.path.join(train_dataset, 'image')))
    training_imgs = len(train_files)
    train_ids = list(range(1, training_imgs + 1))
    
    X_train = np.zeros((len(train_ids), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_SIZE, IMG_SIZE, 1), dtype=np.bool)
    
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        X_train[n] = imread(train_dataset + '/image/' + str(id_) + '.png')
        mask = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.bool)
        for mask_file in next(os.walk(train_dataset  + '/mask/')):
            mask_ = imread(train_dataset + '/mask/' + str(id_) + '.png')
            mask_ = np.expand_dims(mask_, axis=-1)
            mask = np.maximum(mask, mask_)
        
        Y_train[n] = mask
    
    _, test_fol, _ = next(os.walk(test_dataset))
    _, _, test_files = next(os.walk(os.path.join(test_dataset, test_fol[0], 'image')))
    test_imgs = len(test_files)
    test_ids = list(range(1, test_imgs + 1))
    
    X_test = {}
    Y_test = {}
    
    for i in test_fol:
        X_test[i] = np.zeros((len(test_ids), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        Y_test[i] = np.zeros((len(test_ids), IMG_SIZE, IMG_SIZE, 1), dtype=np.bool)
        
    for i in test_fol:
        test_path = os.path.join(test_dataset, i)
        for n, id_ in tqdm(enumerate(test_files), total=len(test_files)):
            X_test[i][n] = imread(test_path + '/image/' + str(id_))
            mask = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.bool)
            for mask_file in next(os.walk(test_path  + '/mask/')):
                mask_ = imread(test_path + '/mask/' + str(id_))
                mask_ = np.expand_dims(mask_, axis=-1)
                mask = np.maximum(mask, mask_)
            
            Y_test[i][n] = mask
    
    return X_train, Y_train, X_test, Y_test

    