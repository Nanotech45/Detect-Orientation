# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 10:16:44 2017

@author: kmcfall
"""
from sklearn.preprocessing import StandardScaler
import numpy as np

x = np.column_stack(([5,4,8,4,2,1,6,9],[8,5,10,67,105,234,54,-78]))
w = np.zeros((1,2))
y = np.array([5,4])
w[0,:] = y
yP = y.reshape(1,-1)
yQ = y.reshape(-1,1)
print(y)
print(w)
print(yP)
print(yQ)
scaler = StandardScaler()
scaler.fit(x)
xScaled = scaler.transform(x)
print(xScaled)
print(" ")
print(np.mean(xScaled[:,0]),np.std(xScaled[:,0]))
print(np.mean(xScaled[:,1]),np.std(xScaled[:,1]))
print(scaler.transform(yP))