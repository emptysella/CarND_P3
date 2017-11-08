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
import matplotlib.pyplot as plt


# Read cvs and get samples
def readCSVData(csvPath):
    samples = []
    with open(csvPath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples

# This function balance de data set. Redux de number of sample
# for a steerin angle range
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


csvPath = 'driving_log.csv'
samples = readCSVData(csvPath)
samplesBalanced = BalanceSamples(samples, 'Range' , 0.001)

steerings = []
for sample in samplesBalanced:
    steerings.append(float(sample[3]))

# Plot histogram
Y = np.array(steerings)
histogram=plt.figure()
hist,bins = np.histogram(Y, 800 , [-1.0,1.0])
plt.hist(Y, bins, alpha=0.5)
plt.show()
