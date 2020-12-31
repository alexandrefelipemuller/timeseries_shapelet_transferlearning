#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 20:11:19 2016

@author: stephen
"""
 
from __future__ import print_function
 
from keras.models import Model
from keras.utils import np_utils
import numpy as np
import os
from keras.callbacks import ModelCheckpoint
import pandas as pd
import sys
import keras 
from keras.callbacks import ReduceLROnPlateau
      
def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y
  
nb_epochs = 300


#flist = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 
#'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', '50words', 'FISH', 'Gun_Point', 'Haptics', 
#'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT', 'MedicalImages', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 
#'NonInvasiveFatalECG_Thorax2', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'SwedishLeaf', 'Symbols', 
#'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga']

flist  = [ sys.argv[1] ] 
for each in flist:
    fname = each
    x_train, y_train = readucr(fname+'/'+fname+'_TRAIN')
    x_test, y_test = readucr(fname+'/'+fname+'_TEST')
    nb_classes = len(np.unique(y_test))
    batch_size = int(min(x_train.shape[0]/10, 16))
    y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
    y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)
     
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean)/(x_train_std)
     
    x_test = (x_test - x_train_mean)/(x_train_std)
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    print ("class:"+each+", number of classes: "+str(nb_classes))
    x = keras.layers.Input(x_train.shape[1:])

#    drop_out = Dropout(0.2)(x)
    conv1 = keras.layers.Conv1D(filters=32, kernel_size=8, strides=1, activation='relu', input_shape=(32,1))(x)
    conv1 = keras.layers.normalization.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation('relu')(conv1)
    
#    drop_out = Dropout(0.2)(conv1)
    conv2 = keras.layers.Conv1D(filters=64, kernel_size=5, border_mode='same')(conv1)
    conv2 = keras.layers.normalization.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)
    
#    drop_out = Dropout(0.2)(conv2)
    conv3 = keras.layers.Conv1D(filters=32, kernel_size=3, border_mode='same')(conv2)
    conv3 = keras.layers.normalization.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)
    
    full = keras.layers.pooling.GlobalAveragePooling1D()(conv3)    
    out = keras.layers.Dense(nb_classes, activation='softmax')(full)
    
    
    model = Model(input=x, output=out)
     
    optimizer = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
     
    reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                      patience=50, min_lr=0.0001)
    model.load_weights(fname+'_no_shapelet_best.hdf5')

    checkpointer = ModelCheckpoint(filepath=fname+"_trash.hdf5",
            monitor = 'val_accuracy',
            verbose=2,
            save_best_only=True)

#    hist = model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
#            verbose=1, callbacks=[reduce_lr], validation_data=(x_test, Y_test))
    hist = model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
            verbose=1, callbacks=[checkpointer,reduce_lr], validation_data=(x_test, Y_test))


   #Print the testing results which has the lowest training loss.
    log = pd.DataFrame(hist.history)
    print (log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin])

