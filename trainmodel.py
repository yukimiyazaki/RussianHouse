#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
based on fcn semantic segmentation of Long
and
body part segmentation
1222sec/epoch
"""
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Reshape, Flatten, Dropout, TimeDistributed, Input, merge, GaussianNoise, BatchNormalization
from keras.layers import LSTM
from keras.layers import Convolution2D, Deconvolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.objectives import categorical_crossentropy
from keras.regularizers import l2
from keras.models import model_from_yaml
from keras.utils import np_utils
# from keras.initializations import normal, zero
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.applications.vgg16 import VGG16


import pickle
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

startTime = time.time()


def model(input_shape, dimension_list):
    #input_shapeの形をinputとして扱う
    input_layer = Input(shape=input_shape)
#dropoutいるかも
    # Block 1
    x = Dense(dimension_list[0], activation='linear')(input_layer)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
#x = Dropout(0.5)(x)
    # Block 2
    x = Dense(dimension_list[1], activation='linear')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
# x = Dropout(0.5)(x)
    # Block 3
    x = Dense(dimension_list[2], activation='linear')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
# x = Dropout(0.5)(x)
    # Block 4
    x = Dense(1, activation='linear')(x)
    x = BatchNormalization()(x)
    y = Activation(activation='relu')(x)
    model = Model(input=input_layer, output=y)
    return model


def batch_generator(datapath='train_std.pkl', batchsize=128, step=128, new_shape=[64,64]):
    with open(datapath, mode='rb') as f:
        data = pickle.load(f)

    label = data[:,277]
    idx = 0
    input = np.zeros([data.shape[0],data.shape[1]-1], dtype=np.float32)
    input[:,:277]  = data[:,:277]
    input[:,277:]  = data[:,278:]

    while True:
        if idx == 0:
            perm1 = np.arange(batchsize * step)
            np.random.shuffle(perm1)

        batchx = input[perm1[idx:idx + batchsize]]
        batchy = label[perm1[idx:idx + batchsize]]
        # print(batchx1.shape)
        # print(batchx2.shape)
        # print(batchy.shape)
        yield batchx, batchy

        if idx + batchsize >= batchsize * step:
            idx = 0
        elif idx + batchsize >= input.shape[0]:
            idx = 0
        else:
            idx += batchsize


# def train():
# parameters
BATCHSIZE = 128
NUM_DATA = 30471 #Todo スマートじゃない
EPOCH = 1000

num_batches = int(NUM_DATA / BATCHSIZE)
datapath = 'train_zvalue.pkl'
gen = batch_generator(datapath=datapath, batchsize=BATCHSIZE, step=num_batches, )

# build model
model = model(input_shape = [455,], dimension_list= [100, 100, 100])
#f_opt = Adam(lr=1e-6, beta_1=0.99)
f_opt = Adam()
#Todo loss functionを考える->binary_crossentropyは負の値になる inputが「01]でないせい？
model.compile(loss='mean_squared_error', optimizer=f_opt)


# train
# next(gen)
history = model.fit_generator(gen, samples_per_epoch=BATCHSIZE * num_batches, nb_epoch=EPOCH)

# save
savepath = 'model/fc1e0'
json_string = model.to_json()
with open(savepath+'.json', "w") as f:
    f.write(json_string)
model.save_weights(savepath+'_W.hdf5')