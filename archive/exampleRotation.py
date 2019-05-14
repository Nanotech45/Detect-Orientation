# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:24:38 2017

@author: kmcfall
"""

import cv2
import matplotlib.pyplot as plot
im = plot.imread('Roomba02.jpg')
x0 = 50
y0 = 100
#                            center   angle  scale
M = cv2.getRotationMatrix2D((x0,y0),  -30,    1  ) 
print(M)
#                    input  matrix        output shape
rot = cv2.warpAffine( im,    M,    (im.shape[1],im.shape[0]))
plot.figure(1)
plot.clf()
plot.imshow(im)
plot.plot(x0,y0,'m*')
plot.figure(2)
plot.clf()
plot.imshow(rot)
plot.plot(x0,y0,'m*')
