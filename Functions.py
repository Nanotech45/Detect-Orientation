# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:41:27 2017

@author: whowland
"""

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

def scaleFeatures(trainingData):
    features = getFeatureValues(trainingData)
    scaler = StandardScaler()
    scaler.fit(features)
    scaled = scaler.transform(features)
    return scaled

def trainClassifier(features, labels, es, hl):
    ann = MLPRegressor(hidden_layer_sizes = (hl,),activation='logistic',solver='lbfgs', early_stopping=es)
    ann.fit(features.T, labels)
    return ann

def testClassifier(features, classifier):
    anglePredictions = []
    for i in range(len(features.T)):
        predictions = classifier.predict(features[:,i].reshape(1,-1))
        x = predictions[0,0]
        y = predictions[0,1]
        anglePredictions.append(XYToDegree(x,y))
    return anglePredictions

def getFeatureValues(rawData):
    for i in range(len(rawData)):
        hsv = cv2.cvtColor(rawData[i], cv2.COLOR_RGB2HSV)
        featHSV = hog(hsv[:,:,1], orientations=5, pixels_per_cell=(7, 7), cells_per_block=(2, 2), block_norm='L1-sqrt', visualise=False)
        edge = cv2.Canny(cv2.cvtColor(rawData[i], cv2.COLOR_RGB2GRAY),100,300)
        featEdge = hog(edge, orientations=5, pixels_per_cell=(7, 7), cells_per_block=(2, 2), block_norm='L1-sqrt', visualise=False)
        if (i == 0):
            features = np.concatenate((featHSV, featEdge))
        else:
            features = np.column_stack((features, np.concatenate((featHSV, featEdge))))
    return features

def degreeToXY(degree):
    toRad = np.pi / 180
    return (np.cos(degree*toRad),np.sin(degree*toRad))

def XYToDegree(x,y):
    toDeg = 180 / np.pi
    degree = np.arctan2(y,x)*toDeg
    if degree < 0:
        degree += 360
    return degree

def calculateMAEDeg(anglePredictions, labels):
    angleError = []
    for i in range (int(len(labels))):
        error = abs(labels[i] - anglePredictions[i])
        if (error >= 180):
            error -= 360
        angleError.append(abs(error))
    return angleError


# =============================================================================
#     if trainANNWithPresetES:
#         ANN = Functions.trainClassifier(Functions.scaleFeatures(rotated), angles, True, 10)
# =============================================================================

