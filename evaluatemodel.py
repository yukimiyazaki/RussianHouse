#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VGGの学習済み重みを使わない
"""
from keras import backend as K
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Activation, Reshape, Flatten, Dropout, TimeDistributed, Input, merge, GaussianNoise, BatchNormalization
from keras.layers import LSTM
from keras.layers import Convolution2D, Deconvolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.objectives import categorical_crossentropy
from keras.regularizers import l2
from keras.models import model_from_yaml
from keras.utils import np_utils
from keras.initializations import normal, zero
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.applications.vgg16 import VGG16


import pickle
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
'''
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
'''
# parameters
EPOCH = 1
BATCHSIZE = 7762
NUM_DATA = 7762
num_batches = int(NUM_DATA / BATCHSIZE)
# load model
loadpath = 'model/fc1e0'
f = open(loadpath+'.json', 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string)
train_model.load_weights(loadpath+'_W.hdf5')





with open('train_mean.pkl', mode='rb') as f:
    train_mean = pickle.load(f)

with open('train_std.pkl', mode='rb') as f:
    train_std = pickle.load(f)

datapath = 'test_zvalue.pkl'
with open(datapath, mode='rb') as f:
    testdata = pickle.load(f)

y = train_model.predict(testdata)

prediction = y * train_std[277] + train_mean[277]
prediction = np.round(prediction).astype(int)[:,0]
np.savetxt('prediction20170506_dim_100_100_100_epoch_1000.csv',prediction,)