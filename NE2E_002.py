"""
Date:       march 2017
Author:     Sergio Valero | Udacity Self-Driving Car Nano Degree
"""

import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Lambda,Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D

# Reduced arquitecture with dropout
def NE2E(param):

    model = Sequential()

    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(param['row'], param['col'], param['ch'])))
    model.add(Cropping2D(cropping=((param['cropUp'],param['cropDown']), (0,0))))

    model.add(Convolution2D(24, 5, 5, subsample=(2,2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(36, 5, 5, subsample=(2,2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Convolution2D(48, 5, 5, subsample=(2,2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dense(1))

    model.compile(loss=param['loss'], optimizer='adam')

    return model

# Compilation
#inputParam = {'ch':3, 'row':160, 'col':320,'cropUp':70,'cropDown':20,}
#model = NE2E(inputParam)
