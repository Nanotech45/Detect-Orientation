# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 07:41:57 2017

@author: kmcfall
"""

# Part 1 - basic
import numpy as np
import matplotlib.pyplot as plot
x = np.linspace(0,10,10)
y = np.polyval([0.5,0],x)
w = np.random.rand(1)
mu = 0.005
plot.figure(2)
E = 100
count = 0
while abs(E)>1e-7:
    print(w)
    count += 1
    E = np.sum(y - w*x)
    w += mu*np.sum(y - w*x)
    plot.clf()
    plot.plot(x, y  ,'m*')
    plot.plot(x, w*x,'b' )
    plot.ylim((-1,6))
    plot.xlabel('x')
    plot.ylabel('y')
    plot.pause(0.001)
print(count)


'''
# Part 2 - bias
import numpy as np
import matplotlib.pyplot as plot
x = np.linspace(0,10,10)
y = np.polyval([0.3,1],x)
w1 = np.random.rand(1)
w0 = np.random.rand(1)
mu = 0.005
plot.figure(2)
E = 100
count = 0
while abs(E)>1e-7:
    print(w1,w0)
    count += 1
    E = np.sum(y - (w1*x+w0))
    w0 += mu*np.sum(y - (w1*x+w0))
    w1 += mu*np.sum(y - (w1*x+w0))
    plot.clf()
    plot.plot(x, y  ,'m*')
    plot.plot(x, w1*x+w0,'b' )
    plot.ylim((-1,6))
    plot.xlabel('x')
    plot.ylabel('y')
    plot.pause(0.001)
print(count)
'''

'''
# Part 3 - improved learning
import numpy as np
import matplotlib.pyplot as plot
x = np.linspace(0,10,10)
y = np.polyval([0.3,1],x)
w1 = np.random.rand(1)
w0 = np.random.rand(1)
mu = 0.005
plot.figure(2)
E = 100
count = 0
while abs(E)>1e-7:
    print(w1,w0)
    count += 1
    E = np.sum(y - (w1*x+w0))
    w0 += mu*np.sum(   y - (w1*x+w0))
    w1 += mu*np.sum(x*(y - (w1*x+w0)))
    plot.clf()
    plot.plot(x, y  ,'m*')
    plot.plot(x, w1*x+w0,'b' )
    plot.ylim((-1,6))
    plot.xlabel('x')
    plot.ylabel('y')
    plot.pause(0.001)
print(count)
'''

'''
# Part 4 (gradient descent)
import sympy
import numpy as np
import matplotlib.pyplot as plot
x = np.linspace(0,10,10)
y = np.polyval([0.3,1],x)
w1 = np.random.rand(1)
w0 = np.random.rand(1)
mu = 0.001
plot.figure(2)
count = 0
E = 100
w0Sym, w1Sym = sympy.symbols('w0 w1')
Esym = sympy.sympify(0)
for i in range(x.shape[0]):
    Esym += sympy.sympify((y[i] - (w1Sym*x[i]+w0Sym))**2)
while abs(E)>1e-7:
    print(w1,w0)
    count += 1
    E = Esym.subs([(w0Sym,w0),(w1Sym,w1)])
    w0 -= mu*float(sympy.diff(Esym,w0Sym).subs([(w0Sym,w0),(w1Sym,w1)]))
    w1 -= mu*float(sympy.diff(Esym,w1Sym).subs([(w0Sym,w0),(w1Sym,w1)]))
    plot.clf()
    plot.plot(x, y  ,'m*')
    plot.plot(x, w1*x+w0,'b' )
    plot.ylim((-1,6))
    plot.xlabel('x')
    plot.ylabel('y')
    plot.pause(0.001)
print(count)
'''

'''
# Linear fails with parabola
import sympy
import numpy as np
import matplotlib.pyplot as plot
x = np.linspace(0,10,10)
y = np.polyval([1,-8,4],x)
w1 = np.random.rand(1)
w0 = np.random.rand(1)
mu = 0.001
plot.figure(2)
count = 0
E = 100
w0Sym, w1Sym = sympy.symbols('w0 w1')
Esym = sympy.sympify(0)
for i in range(x.shape[0]):
    Esym += sympy.sympify((y[i] - (w1Sym*x[i]+w0Sym))**2)
print(Esym)
while count < 100:
    E = Esym.subs([(w0Sym,w0),(w1Sym,w1)])
    print(w1,w0)
    count += 1
    w0 = w0 - mu*sympy.diff(Esym,w0Sym).subs([(w0Sym,w0),(w1Sym,w1)])
    w1 = w1 - mu*sympy.diff(Esym,w1Sym).subs([(w0Sym,w0),(w1Sym,w1)])
    plot.clf()
    plot.plot(x, y  ,'m*')
    plot.plot(x, w1*x+w0,'b' )
    plot.ylim((-23,34))
    plot.xlabel('x')
    plot.ylabel('y')
    plot.pause(0.001)
print(count)
'''

'''
# Sigmoid transfer function
import sympy
import numpy as np
import matplotlib.pyplot as plot
def fSym(a):
    return 1/(1 + sympy.exp(-a))
def f(a):
    return 1/(1 +    np.exp(-np.float16(a)))
x = np.linspace(0,10,10)
xDense = np.linspace(0,10,100)
y = np.polyval([1,-8,4],x)
w1 = np.random.rand(1)
w0 = np.random.rand(1)
mu = 0.5
plot.figure(2)
count = 0
E = 100
w0Sym, w1Sym = sympy.symbols('w0 w1')
Esym = sympy.sympify(0)
for i in range(x.shape[0]):
    Esym += sympy.sympify((y[i] - fSym(w1Sym*x[i]+w0Sym))**2)
while count < 100:
    E = Esym.subs([(w0Sym,w0),(w1Sym,w1)])
    print(w1,w0)
    count += 1
    w0 = w0 - mu*sympy.diff(Esym,w0Sym).subs([(w0Sym,w0),(w1Sym,w1)])
    w1 = w1 - mu*sympy.diff(Esym,w1Sym).subs([(w0Sym,w0),(w1Sym,w1)])
    plot.clf()
    plot.plot(x, y  ,'m*')
    plot.plot(xDense, f(w1*xDense+w0),'b' )
    plot.ylim((-23,34))
    plot.xlabel('x')
    plot.ylabel('y')
    plot.pause(0.001)
print(count)
'''

'''
# MLP
import sympy
import numpy as np
import matplotlib.pyplot as plot
def fSym(a):
    return 1/(1 + sympy.exp(-a))
def f(a):
    return 1/(1 +    np.exp(-a))
xDense = np.linspace(0,10,100)
x = np.linspace(0,10,10)
y = np.polyval([1,-8,4],x)
w01 = np.random.rand(1)
w00 = np.random.rand(1)
w11 = np.random.rand(1)
w10 = np.random.rand(1)
u2 = np.random.rand(1)
u1 = np.random.rand(1)
u0 = np.random.rand(1)
mu = 0.00025
plot.figure(2)
count = 0
E = 100
w00Sym, w01Sym, w10Sym, w11Sym, u0Sym, u1Sym, u2Sym = sympy.symbols('w00 w01 w10 w11 u0 u1 u2')
Esym = sympy.sympify(0)
for i in range(x.shape[0]):
    Esym += sympy.sympify((y[i] - (u0Sym + u1Sym*fSym(w01Sym*x[i]+w00Sym) + u2Sym*fSym(w11Sym*x[i]+w10Sym)))**2)
while abs(E)>1e-7:
    E = Esym.subs([(w00Sym,w00),(w01Sym,w01),(w10Sym,w10),(w11Sym,w11),(u0Sym,u0),(u1Sym,u1),(u2Sym,u2)])
    count += 1
    w00 -= mu*float(sympy.diff(Esym,w00Sym).subs([(w00Sym,w00),(w01Sym,w01),(w10Sym,w10),(w11Sym,w11),(u0Sym,u0),(u1Sym,u1),(u2Sym,u2)]))
    w01 -= mu*float(sympy.diff(Esym,w01Sym).subs([(w00Sym,w00),(w01Sym,w01),(w10Sym,w10),(w11Sym,w11),(u0Sym,u0),(u1Sym,u1),(u2Sym,u2)]))
    w10 -= mu*float(sympy.diff(Esym,w10Sym).subs([(w00Sym,w00),(w01Sym,w01),(w10Sym,w10),(w11Sym,w11),(u0Sym,u0),(u1Sym,u1),(u2Sym,u2)]))
    w11 -= mu*float(sympy.diff(Esym,w11Sym).subs([(w00Sym,w00),(w01Sym,w01),(w10Sym,w10),(w11Sym,w11),(u0Sym,u0),(u1Sym,u1),(u2Sym,u2)]))
    u0  -= mu*float(sympy.diff(Esym, u0Sym).subs([(w00Sym,w00),(w01Sym,w01),(w10Sym,w10),(w11Sym,w11),(u0Sym,u0),(u1Sym,u1),(u2Sym,u2)]))
    u1  -= mu*float(sympy.diff(Esym, u1Sym).subs([(w00Sym,w00),(w01Sym,w01),(w10Sym,w10),(w11Sym,w11),(u0Sym,u0),(u1Sym,u1),(u2Sym,u2)]))
    u2  -= mu*float(sympy.diff(Esym, u2Sym).subs([(w00Sym,w00),(w01Sym,w01),(w10Sym,w10),(w11Sym,w11),(u0Sym,u0),(u1Sym,u1),(u2Sym,u2)]))
    plot.clf()
    plot.plot(x, y  ,'m*',label='training data')
    plot.plot(xDense, u1*f(w01*xDense+w00),'c',label='node 0')
    plot.plot(xDense, u2*f(w11*xDense+w10),'g', label='node 1')
    plot.plot(xDense, u0 + u1*f(w01*xDense+w00) + u2*f(w11*xDense+w10),'b' ,label = 'final output')
    plot.ylim((-23,34))
    plot.xlabel('x')
    plot.ylabel('y')
    plot.legend()
    plot.pause(0.001)
print(count)
'''

'''
# Using sklearn
from sklearn.neural_network import MLPRegressor as MLP
import numpy as np
import matplotlib.pyplot as plot
h = 10
xDense = np.linspace(0,10,100).reshape((-1,1)) # Must be column vector
x = np.linspace(0,10,10).reshape((-1,1))       # Must be column vector
y = np.polyval([1,-8,4],x).ravel()             # Prefers 1-D array over column vector
ANN = MLP(hidden_layer_sizes = (h,),activation='logistic',solver='lbfgs')
ANN.fit(x,y)
plot.figure(2)
plot.clf()
plot.plot(x, y  ,'m*')
plot.plot(xDense,ANN.predict(xDense))
plot.xlabel('x')
plot.ylabel('y')
'''

'''
# Using sklearn - multiple outputs
from sklearn.neural_network import MLPRegressor as MLP
import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
x,y = np.meshgrid(np.linspace(0,np.pi/2),
                  np.linspace(-np.pi/2,np.pi/2))
z =     [np.sin(x*y)]
z.append(np.exp(x-y))
z.append(np.ones_like(x))
feat   = np.column_stack(( x.reshape(-1,1),
                           y.reshape(-1,1)))
labels = np.column_stack((z[0].reshape(-1,1),
                          z[1].reshape(-1,1),
                          z[2].reshape(-1,1)))
ANN = MLP(hidden_layer_sizes = (4,),
          activation='logistic',solver='lbfgs')
ANN.fit(feat,labels)
N = ANN.predict(feat)
N1 = N[:,0].reshape(50,50)
N2 = N[:,1].reshape(50,50)
N3 = N[:,2].reshape(50,50)
fig = plot.figure(1)
plot.clf()
for i in range(3):
    ax = plot.subplot(3,2,2*i+1,projection='3d')
    ax.plot_surface(x,y,z[i])
    ax = plot.subplot(3,2,2*i+2,projection='3d')
    ax.plot_surface(x,y,N[:,i].reshape(50,50))
ax.set_zlim(0,2)
print('feature matrix',feat.shape)
print('  label matrix',labels.shape)
print(' layer weights',ANN.coefs_[0].shape)
print('  layer biases',ANN.intercepts_[0].shape)
print('output weights',ANN.coefs_[1].shape)
print(' output biases',ANN.intercepts_[1].shape)
print('   ANN outputs',N.shape)
'''