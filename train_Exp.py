"""
Date:       march 2017
Author:     Sergio Valero | Udacity Self-Driving Car Nano Degree
"""


import csv
import numpy as np
import cv2
import sklearn
import random
import pickle
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Lambda,Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger, History
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from NE2E_002 import NE2E



# csvPath = './data/driving_log.csv'
def readCSVData(csvPath):
    samples = []
    with open(csvPath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples

def BalanceSamples(samples, mode , reduxFactor):
    samplesOut = []
    if (mode == 'Range'):
        for sample in samples:
            steeringRand = random.randint(0, 1)
            steeringRand2 = random.randint(0, 1)
            if ((float(sample[3]) > -(reduxFactor) ) & (float(sample[3]) < reduxFactor)):
                if (steeringRand & steeringRand2):
                    samplesOut.append(sample)
            else:
                samplesOut.append(sample)
    elif (mode == 'All'):
        samplesOut = samples

    return samplesOut


"""
Generator to train with L,C,R camera images and Random aumentation flip mages
"""
def generator(samples, batch_size=8, camNum=3, factor=0.20):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        # process total sample by baches
        for offset in range(0,num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            steerings = []
            # process every sample in a bach
            for batch_sample in batch_samples:

                flipRand = random.randint(0,1)
                steeringRand = random.randint(0,1)
                steering = float(batch_sample[3]) if flipRand else -float(batch_sample[3])
                correctionFactor  = factor
                # Read all images for each sample and Compute augmentation fliping images
                for i in range(camNum):
                    source_path = batch_sample[i]
                    tokens = source_path.split('/')
                    filename = tokens[-1]
                    local_path = "./data/IMG/" + filename
                    image = cv2.imread(local_path) if flipRand else cv2.flip(cv2.imread(local_path),1)
                    assert image is not None
                    images.append(image)
                    if i==0:steerings.append(steering) # Steering angle label for Center cam. image
                    if i==1:steerings.append(steering + correctionFactor) # Steering angle label for Left cam. image
                    if i==2:steerings.append(steering - correctionFactor) # Steering angle label for Right cam. image
            X_train = np.array(images)
            y_train = np.array(steerings)
            yield sklearn.utils.shuffle(X_train, y_train)

"""
Here we start to train the model
"""
csvPath = './data/driving_log.csv'
samples = readCSVData(csvPath)
modeB = 'Range' # ['All', 'Range']
inputParam = {'ch':3, 'row':160, 'col':320,'cropUp':60,'cropDown':1, 'loss':'mse'} # 'loss':['mse','mae']
bs = 256
ne = 100
camNumTrain = 3
camNumTest = 1
fac = 0.10
arq = 'NE2E_002'

# Samples manager
samplesB = BalanceSamples(samples, modeB , 0.001)
random_state_1 = random.randint(0, 100)
random_state_2 = random.randint(0, 100)
Samples50, notUsedSamples = train_test_split(samplesB, test_size=0.7, random_state=random_state_1)
train_samples, validation_samples = train_test_split(Samples50, test_size=0.3, random_state=random_state_2)

model = NE2E(inputParam)

for i in range(10,12,2) : # Iteraror to tune the camera angle factor
    fac = i*0.01
    nameLabel = 'MDL_' + str(len(train_samples)*2) + '_NE_' + str(ne) + '_LSS_' + inputParam['loss'] + '_' + arq + '_MD_' + modeB + '_F_' + str(int(fac*100)) + '_CAM_' + str(camNumTrain)
    csvName = nameLabel + '.csv'
    modelName = nameLabel + '.h5'
    modelEnd = nameLabel + '_END.h5'
    # Call the model
    model = NE2E(inputParam)
    csv_logger = CSVLogger(csvName)
    checkpointer = ModelCheckpoint(filepath=modelName, verbose=1, save_best_only=True)
    #Training the model
    history_object = model.fit_generator(generator(train_samples, batch_size=bs, camNum=camNumTrain, factor = fac),
                        samples_per_epoch = len(train_samples)*1, validation_data = generator(validation_samples,
                        batch_size=bs, camNum=camNumTest, factor = fac), nb_val_samples=len(validation_samples),
                        nb_epoch=ne,verbose=1,callbacks=[csv_logger, checkpointer])

    model.save(modelEnd)


### print the keys contained in the history object
print(history_object.history.keys())
