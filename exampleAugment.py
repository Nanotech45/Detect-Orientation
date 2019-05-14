# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:08:00 2017

@author: kmcfall
"""
import numpy as np
import matplotlib.pyplot as plot
import skimage.color as color

H = 3.5 # ft hegiht of camera above road
psi = 50*np.pi/180 # 50° diagonal angle of view


def shiftCamera(road,D):
    Nr = int(road.shape[0])
    Nc = int(road.shape[1])    
    NcNew = int(2.15*Nc);
    roadNew = np.zeros((Nr,NcNew));
    roadNew[0:int(Nr/2),int(NcNew/2-Nc/2):int(NcNew/2+Nc/2)] = road[0:int(Nr/2),:];
    weight  = np.ones((Nr,NcNew))
    weight[int(Nr/2):Nr,:] = 0;
    for row in range(int(Nr/2),Nr-2):
        y = row - (Nr/2+0.5)
        for col in range(Nc-1):
            x = col - (Nc/2+0.5)
            xNew = D/H*y + x
            yNew = y
            rowNew = yNew + (Nr   /2+0.5)
            colNew = xNew + (NcNew/2+0.5)
            rowFrac = 1 - (rowNew - np.floor(rowNew))
            colFrac = 1 - (colNew - np.floor(colNew))
            rowNew = int(np.floor(rowNew))
            colNew = int(np.floor(colNew))
            mat = np.array([ [   rowFrac +colFrac ,    rowFrac +(1-colFrac)], 
                             [(1-rowFrac)+colFrac , (1-rowFrac)+(1-colFrac)] ])/4
            roadNew[rowNew:rowNew+2,colNew:colNew+2] = roadNew[rowNew:rowNew+2,colNew:colNew+2] + road[row,col]*mat
            weight [rowNew:rowNew+2,colNew:colNew+2] = weight [rowNew:rowNew+2,colNew:colNew+2] + mat
    weight[weight==0] = 1
    return roadNew/weight
    
def rotateCamera(road,theta,Nr,Nc):
    NcNew = int(2.34*Nc)
    f = np.sqrt(Nr**2 + Nc**2)/2/np.tan(psi/2)
    roadNew = np.zeros((Nr,NcNew));
    weight  = np.zeros((Nr,NcNew))
    for row in range(0,Nr-1):
        y = row - (Nr/2+0.5)
        for col in range(road.shape[1]-1):
            x = col - (road.shape[1]/2+0.5)
            xNew = (x*np.cos(theta) - f*np.sin(theta))/(x/f*np.sin(theta) + np.cos(theta));
            yNew =  y                                 /(x/f*np.sin(theta) + np.cos(theta));            
            rowNew = yNew + (Nr   /2+0.5)
            colNew = xNew + (NcNew/2+0.5)
            rowFrac = 1 - (rowNew - np.floor(rowNew))
            colFrac = 1 - (colNew - np.floor(colNew))
            rowNew = int(np.floor(rowNew))
            colNew = int(np.floor(colNew))
            mat = np.array([ [   rowFrac +colFrac ,    rowFrac +(1-colFrac)], 
                             [(1-rowFrac)+colFrac , (1-rowFrac)+(1-colFrac)] ])/4
            if rowNew >=0 and rowNew+2<Nr and colNew>=0 and colNew+2<NcNew:
                roadNew[rowNew:rowNew+2,colNew:colNew+2] = roadNew[rowNew:rowNew+2,colNew:colNew+2] + road[row,col]*mat
                weight [rowNew:rowNew+2,colNew:colNew+2] = weight [rowNew:rowNew+2,colNew:colNew+2] + mat
    weight[weight==0] = 1
    return roadNew/weight

im = plot.imread('roadScene.jpg')
road = color.rgb2gray(im)
Nr = int(road.shape[0])
Nc = int(road.shape[1])    
plot.figure(1)
plot.imshow(road,cmap='gray')
plot.plot([0,0,Nc-1,Nc-1,0],[0,Nr-1,Nr-1,0,0],'m',linewidth=5)

roadShift = shiftCamera(road,5) # Shift camera 5 ft to the left
NcNew = roadShift.shape[1]
plot.figure(3)
plot.clf()
plot.imshow(roadShift,cmap='gray')
plot.plot([int(NcNew/2-Nc/2),int(NcNew/2-Nc/2),int(NcNew/2+Nc/2),int(NcNew/2+Nc/2),int(NcNew/2-Nc/2)],[0,Nr-1,Nr-1,0,0],'m',linewidth=4)

roadRotate = rotateCamera(roadShift,20*np.pi/180,Nr,Nc) # Rotate camera 20° back
NcNew = roadRotate.shape[1]
plot.figure(4)
plot.clf()
plot.imshow(roadRotate,cmap='gray')
plot.plot([int(NcNew/2-Nc/2),int(NcNew/2-Nc/2),int(NcNew/2+Nc/2)-1,int(NcNew/2+Nc/2)-1,int(NcNew/2-Nc/2)],[0,Nr-1,Nr-1,0,0],'m',linewidth=4)