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
import tensorflow as tf
from tensorflow.data import Dataset
from typing import List, Tuple
import random

OGIMG_SIZE = 1500
IMG_SIZE = 224
OVERLAP = 14
AUTOTUNE = tf.data.experimental.AUTOTUNE

def imgstitch(img_path):
    """
    This function overlays the predicted image patches with each other to produce the final output image.
    It should be noted that the overlap regions are INTENDED to be averaged to minimize error. This is not
    done in this function. They are just overwritten by the next image patch. This will be implemented in the future 
    for a more robust approach
    
    Parameters
    ----------
    img_path : STRING
        Path to directory with test image patches
    Returns
    -------
    None. The final stitched image will be automatically saved in the same directory as the image patches with the 
    name 'ouptut.png''
    """
    _, _, img_files = next(os.walk(img_path))
    
    img_files = sorted(img_files,key=lambda x: int(os.path.splitext(x)[0]))
    IMG_WIDTH, IMG_HEIGHT = (Image.open(img_path + '/11.png')).size
    # IMG_WIDTH, IMG_HEIGHT = IMG_SIZE, IMG_SIZE
    
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
    
def DatasetLoad(train_dataset, test_dataset, val_dataset):
    """
    
    Parameters
    ----------
    train_dataset : STRING
        Sampled training images directory 
    test_dataset : STRING
        Sampled test images directory
    val_dataset : STRING
        Sampled validation images directory
    Returns
    -------
    X_train : NUMPY ARRAY
        Training dataset to be used for features. Outputs a numpy array of size [NUM_OF_SAMPLES, 224, 224, 3]
    Y_train : NUMPY ARRAY
        Training dataset to be used for labels. Outputs a numpy array of size [NUM_OF_SAMPLES, 224, 224, 1]
    X_test : NUMPY ARRAY
         Test dataset to be used for feature predictions. Outputs a numpy array of size [NUM_OF_SAMPLES, 224, 224, 3]
    Y_test : NUMPY ARRAY
         Test dataset to be used for label prediction. Outputs a numpy array of size [NUM_OF_SAMPLES, 224, 224, 1]
    X_val : NUMPY ARRAY
         Validation dataset to be used for feature validation. Outputs a numpy array of size [NUM_OF_SAMPLES, 224, 224, 3]
    Y_val : NUMPY ARRAY
         Validation dataset to be used for label validation. Outputs a numpy array of size [NUM_OF_SAMPLES, 224, 224, 1]
    """
    
    ### TRAINING DATASET ###
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
    
    if isinstance(val_dataset, str) == True:
        ### VALIDATION DATASET ###
        _, _, val_files = next(os.walk(os.path.join(val_dataset, 'image')))
        val_imgs = len(val_files)
        val_ids = list(range(1, val_imgs + 1))
        
        X_val = np.zeros((len(val_ids), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        Y_val = np.zeros((len(val_ids), IMG_SIZE, IMG_SIZE, 1), dtype=np.bool)
        
        for n, id_ in tqdm(enumerate(val_ids), total=len(val_ids)):
            X_val[n] = imread(val_dataset + '/image/' + str(id_) + '.png')
            mask = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.bool)
            for mask_file in next(os.walk(val_dataset  + '/mask/')):
                mask_ = imread(val_dataset + '/mask/' + str(id_) + '.png')
                mask_ = np.expand_dims(mask_, axis=-1)
                mask = np.maximum(mask, mask_)
            
            Y_val[n] = mask
            
    else:
        X_val = None; Y_val = None
                
        
    ### TESTING DATASET ###
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
    
    return X_train, Y_train, X_test, Y_test, X_val, Y_val

class DatasetLoader(object):
    def __init__(self, image_path: List[str], mask_path: List[str], image_size: Tuple[int], channels: Tuple[int] = (3, 1),
                 seed: int = None):
        '''
        Initialize the dataset loader

        Parameters
        ----------
        image_path : Path to images
        mask_path : Path to masks
        image_size : Tuple of the image size
        channels : Tuple the image channels. The default is (3, 1) with the first element corresponding to the image and the second
        corresponding to the mask

        '''
        
        self.image_path = image_path
        self.mask_path = mask_path
        self.image_size = image_size
        self.channels = channels
        
        if seed is None:
            self.seed = random.randint(0, 1000)
        else:
            self.seed = seed
            
    def _parse(self, image_path, mask_path):
        img = tf.io.read_file(image_path)
        msk = tf.io.read_file(mask_path)
        
        images = tf.image.decode_png(img, channels = self.channels[0])
        masks = tf.image.decode_png(msk, channels = self.channels[1])
        
        return images, masks
    
    @tf.function
    def _map(self, image_path, mask_path):
        image, mask = self._parse(image_path, mask_path)
        
        # There's an issue with ValueError: as_list() is not defined on an unknown TensorShape.
        # Likely caused by py_function not explicitly defining the shapes of the tensors. Tried to use
        # set_shape but didn't work. Look into issues later.
        def _load_func(image_f, mask_f):
            image_f = tf.cast(image_f, tf.float32)
            mask_f = tf.cast(mask_f, tf.bool)
            image_f = image_f.set_shape((self.image_size[0], self.image_size[1], self.channels[0]))
            mask_f = mask_f.set_shape((self.image_size[0], self.image_size[1], self.channels[1]) )
            return image_f, mask_f
        return tf.py_function(_load_func, [image, mask], [tf.float32, tf.bool])
        # return image, mask
    
    def data_batch(self, batch_size, shuffle = False):
        '''
        Takes the input data and reads, normalizes, shuffles, and batches it

        Parameters
        ----------
        batch_size : int
            Number of image-mask pairs to load
        shuffle : bool
            Shuffles the data. The default is False.

        Returns
        -------
        A Tensorflow dataset object

        '''
        
        dataset = Dataset.from_tensor_slices((self.image_path, self.mask_path))
        
        dataset = dataset.map(self._map, num_parallel_calls=AUTOTUNE)
        
        if shuffle:
            dataset = dataset.prefetch(AUTOTUNE). shuffle(random.randint(0, len(self.image_path))).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
            
        return dataset