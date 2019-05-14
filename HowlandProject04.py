# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 05:43:11 2017

@author: Nanotech
"""



import cv2
import time
import glob
import numpy as np
import pickle as rick # Pickle Rick!
import matplotlib.pyplot as plot
import Functions
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split as tts

# *******************************
# ****** PROGRAM VARIABLES ******
getNewRoratedImages     = True
saveImagesAsJpg         = False

trainClassifier         = False
displayFigure2          = False
displayFigure3          = False

makeTestEvalImage       = True
makeArrowImage          = True

printPredictions        = False # WARNING: This spits out a lot of data
# *******************************

# Variables
directoryToMakeImagesFrom = 'Data/test/*.jpg' # CHANGE THIS TO THE DIRECTORY OF THE TEST IMAGE DR McFALL!
size = 200
var = 6
res = 50

# ****************************************************************************

if getNewRoratedImages:
    fname = glob.glob(directoryToMakeImagesFrom)
    labels = []
    angles = []
    rotated = []
    tf = time.time()
    print('Calculating Rotated Images...')
    for f in range(len(fname)):
        im = plot.imread(fname[f])
        rot = cv2.resize(im,(res+var,res+var))
        rotated.append(rot)
        labels.append(f)
        angles.append(np.array([Functions.degreeToXY(f)[0], Functions.degreeToXY(f)[1]]))
# =============================================================================
#         plot.figure(1)
#         plot.clf()
#         plot.imshow(im)
#         plot.text(50,-50,'Please Click Center of Roomba:', size=16, color='b')
#         print('Please Click Center of Roomba:')
#         print('')
#         center = plot.ginput(1, timeout=30)
#         center = np.array(center)
#         points = np.array([[center[:,0]-(size/2)-(var/2),center[:,1]-(size/2)-(var/2)],[center[:,0]+(size/2)+(var/2),center[:,1]+(size/2)+(var/2)]])
#         for i in range(360):
#             mini = cv2.resize(im[int(min(points[:,1])):int(max(points[:,1])),int(min(points[:,0])):int(max(points[:,0]))],(res+var,res+var))
#             rot = cv2.warpAffine(mini,cv2.getRotationMatrix2D((int((res+var)/2),int((res+var)/2)),i,1),(mini.shape[1],mini.shape[0]))
#             rotated.append(rot)
#             labels.append(i)
#             angles.append(np.array([Functions.degreeToXY(i)[0], Functions.degreeToXY(i)[1]]))
#             if saveImagesAsJpg:
#                 plot.imsave(('Data/Train/Roomba' + str(f+1) + '_' + str(i) + '.png'), rot)
#         plot.close()
# =============================================================================
    rick.dump(labels, open("Data/labels.pkl","wb"))
    rick.dump(angles, open("Data/angles.pkl","wb"))
    rick.dump(rotated, open("Data/rotatedImages.pkl","wb"))
    X_train, X_test, l_train, l_test, deg_train, deg_test = tts(Functions.scaleFeatures(rotated).T, angles, labels, test_size=0.2)
    X_train = X_train.T
    X_test = X_test.T
    print('Calculating Rotated Images Took ' + "%.2f" % (time.time() - tf) + ' seconds.')
    print('')
else:
    rotated = rick.load(open("Data/rotatedImages.pkl", "rb"))
    labels = rick.load(open("Data/labels.pkl", "rb"))
    angles = rick.load(open("Data/angles.pkl", "rb"))
    X_train, X_test, l_train, l_test, deg_train, deg_test = tts(Functions.scaleFeatures(rotated).T, angles, labels, test_size=0.2)
    X_train = X_train.T
    X_test = X_test.T

# ****************************************************************************

if trainClassifier:
    print('Training Classifier...')
    tf = time.time()
    maeBig = []
    bestAnnList = []
    averageMAE = []
    numRunsIn = 100
    numRunsOut = 10
    startingHL = 4
    endingHL = 30
    for hl in range (startingHL,endingHL,1):
        maeSmall = []
        for j in range(numRunsOut):
            print('HL:  ' + str(hl) + '\nRun: ' + str(j) + '\n')
            ANN = MLPRegressor(hidden_layer_sizes = (hl,),max_iter=30,activation='logistic',solver='lbfgs',warm_start=True)
            ANN.fit(X_train.T, l_train)
            MAEtrain= []
            MAEval = []
            epochs = []
            annList = []
            bestMAE = float('inf')
            bestMAEIndex = 0
            ANN.fit(X_train.T, l_train)
            for i in range(numRunsIn):
                ANN.fit(X_train.T, l_train)
                annList.append(ANN)
                epochs.append(ANN.n_iter_)
                yPredict = ANN.predict(X_train.T)
                MAEtrain.append(np.mean(np.abs(yPredict[:,1] - np.array(l_train)[:,1]) + np.abs(yPredict[:,0] - np.array(l_train)[:,0])))
                yPredict = ANN.predict(X_test.T)
                MAEval.append(np.mean(np.abs(yPredict[:,1] - np.array(l_test)[:,1]) + np.abs(yPredict[:,0] - np.array(l_test)[:,0])))
                if displayFigure2:
                    plot.figure(2)
                    plot.clf()
                    plot.semilogy(epochs,MAEtrain,label='train')
                    plot.semilogy(epochs,MAEval  ,label='val')
                    plot.legend()
                    plot.pause(0.001)
                if MAEval[-1] < bestMAE:
                    bestMAE = MAEval[-1]
                    bestMAEIndex = i
                    anglePredictions = Functions.testClassifier(X_test, ANN)
                    maeTemp = np.mean(Functions.calculateMAEDeg(anglePredictions, deg_test))
            bestAnnList.append(annList[bestMAEIndex])
            maeSmall.append(maeTemp)
        maeBig.append(maeSmall)
        if displayFigure3:
            plot.figure(3)
            if (hl == 2):
                plot.clf()
            for j in range(len(maeSmall)):
                plot.plot(hl, maeSmall[j], 'rx')
            averageMAE.append(np.average(maeSmall))
            plot.plot(hl, np.average(averageMAE[-1]), 'm*')
            plot.pause(0.001)
    bestHL = np.argmin(averageMAE)+2
    ANN = bestAnnList[bestHL-startingHL]
    rick.dump(ANN, open("Data/ANN.pkl","wb"))
    print('Training Completed in ' + "%.2f" % (time.time() - tf) + ' seconds.')
else:
    ANN = rick.load(open("Data/ANN.pkl", "rb"))

# ****************************************************************************

if makeTestEvalImage:
    print('Creating Error Eval Image...')
    tf = time.time()
    featureMatrix = Functions.scaleFeatures(rotated[0:359])
    anglePredictions = Functions.testClassifier(featureMatrix, ANN)
    maeDeg = Functions.calculateMAEDeg(anglePredictions, labels[0:359])
    highestError = np.max(maeDeg)
    fig = plot.figure(1)
    plot.clf()
    ax = fig.gca()
    plot.plot(labels[0:359], maeDeg[0:359])
    plot.xlabel('Ground truth angle')
    plot.ylabel('Error in angle')
    if printPredictions:
        for i in range(int(len(rotated)/2)):
            print('Actual Angle   : ' + str(labels[i]))
            print('Predicted Angle: ' + str(anglePredictions[i]))
            print()
    anglePredictions = Functions.testClassifier(Functions.scaleFeatures(rotated), ANN)
    mae = np.mean(Functions.calculateMAEDeg(anglePredictions, labels))
    print('Mean Absolute Error for image:    ' + "%.3f" % (mae))
    print('Maximum Absolute Error for image: ' + "%.3f" % (highestError))
    print('Error Eval Image Created in ' + "%.2f" % (time.time() - tf) + ' seconds.')
    
    
# ****************************************************************************

if makeArrowImage:
    print('Creating Arrow Image...')
    tf = time.time()
    anglePredictions = Functions.testClassifier(Functions.scaleFeatures(rotated), ANN)
    if printPredictions:
        for i in range(len(anglePredictions)):
            print('Actual Angle   : ' + str(labels[i]) + ' Degrees')
            print('Predicted Angle: ' + str(anglePredictions[i]) + ' Degrees')
            print()
    fig = plot.figure(2)
    plot.clf()
    ax = fig.gca()
    for i in range(int(len(rotated))):
        dx = Functions.degreeToXY(anglePredictions[i]+90)[1]*int(size/15)
        dy = Functions.degreeToXY(anglePredictions[i]+90)[0]*int(size/15)
        plot.subplot(15,24,i+1)
        plot.imshow(rotated[i])
        plot.arrow(int((res+var)/2), int((res+var)/2), dx, dy, fc='k', ec='k',head_width=4,head_length=6,linewidth=3)
        plot.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    print('Arrow Image Created in ' + "%.2f" % (time.time() - tf) + ' seconds.')





