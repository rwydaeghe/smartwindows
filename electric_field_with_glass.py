#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:34:44 2020

@author: hanne
"""
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, CubicTriInterpolator, LinearTriInterpolator
import numpy as np
import pickle

trifinder = triang.get_trifinder()

tci = LinearTriInterpolator(triang,z)
(Ex,Ey) = tci.gradient(triang.x,triang.y)
E = -np.array([Ex,Ey])

x = np.linspace(0,20e-5,200)
y = np.linspace(y_b - t,y_t + t,y_cells)
X,Y = np.meshgrid(x,y)

def get_field_here(field, pos: np.ndarray) -> np.ndarray:
        tr = trifinder(*pos)                               
        i = triang.triangles[tr]   
        v0 = np.array([triang.x[i[0]],triang.y[i[0]]])  
        v1 = np.array([triang.x[i[1]],triang.y[i[1]]])
        v2 = np.array([triang.x[i[2]],triang.y[i[2]]])
        norm = np.array([np.linalg.norm(v0-pos),np.linalg.norm(v1-pos),np.linalg.norm(v2-pos)])
        j = np.argmin(norm)                                                     
        v = i[j]
        fieldx = np.array(field[0])
        fieldy = np.array(field[1])
        return np.array([fieldx[v], fieldy[v]])

def get_E(pos: np.ndarray=None):
        if pos.any()==None:
            return E
        else:
            return get_field_here(E,pos)

Ex=np.array([get_E(np.array([x,y]))[0] for (x,y) in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
Ey=np.array([get_E(np.array([x,y]))[1] for (x,y) in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

plt.figure()
plt.streamplot(X,Y,Ex,Ey,density=3)
plt.title('Streamplot of Electric Field')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()

plt.figure()
plt.quiver(X,Y,Ex,Ey)
plt.title('Vectorplot of Electric Field')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()




    