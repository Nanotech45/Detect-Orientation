# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 09:16:41 2017

@author: kmcfall
"""

from sklearn.neural_network import MLPRegressor as MLP
import matplotlib.pyplot as plot
import numpy as np
from sklearn.model_selection import train_test_split
x = np.linspace(-5,10,200)
y = np.polyval([0.1, -1, 3, -1],x)
yNoise = y + (np.random.rand(len(x))*8-4)
yFit = np.polyval(np.polyfit(x,yNoise,3),x)
x = x.reshape(-1,1)
ANN = MLP(hidden_layer_sizes = (10,),max_iter=30,activation='logistic',solver='lbfgs',warm_start=True)
xTrain, xVal, yTrain, yVal = train_test_split(x, yNoise, test_size = 0.25)
MAEtrain= []
MAEval = []
epochs = []
numRuns = 200
bestMAE = float('inf')
for i in range(numRuns):
    ANN.fit(xTrain,yTrain)
    epochs.append(ANN.n_iter_)
    yPredict = ANN.predict(xTrain)
    MAEtrain.append(np.mean(np.abs(yPredict - yTrain)))
    yPredict = ANN.predict(xVal)
    MAEval.append(np.mean(np.abs(yPredict - yVal  )))
    yPredict = ANN.predict(x)
    if MAEval[-1] < bestMAE:
        bestMAE = MAEval[-1]
# =============================================================================
#         plot.figure(5)
#         plot.clf()
#         plot.plot(x,y,label='Actual')
#         plot.plot(x,yPredict,label='ANN')
#         plot.plot(xVal  ,yVal  ,'c.',label='val',markersize=2)
#         plot.legend()
#         plot.pause(0.001)
# =============================================================================
    plot.figure(3)
    plot.clf()
    plot.plot(x,y,label='Actual')
    plot.plot(x,yPredict,label='ANN')
    plot.plot(xTrain,yTrain,'m.',label='train',markersize=2)
    plot.plot(xVal  ,yVal  ,'c.',label='val',markersize=2)
    plot.legend()
    plot.pause(0.001)
    plot.figure(4)
    plot.clf()
    plot.semilogy(epochs,MAEtrain,label='train')
    plot.semilogy(epochs,MAEval  ,label='val')
    plot.legend()
    plot.pause(0.001)
