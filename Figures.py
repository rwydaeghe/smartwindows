# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:52:38 2020

@author: Hanne
"""

from scipy.sparse import *
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, CubicTriInterpolator, LinearTriInterpolator
import numpy as np
import pickle
import matplotlib.cm as cm

eps = np.finfo(float).eps 
e   = 1.602e-19
eps_0 = 8.854e-12
eps_r = 2

with open('smartwindows/Variables/triangulation_with_glass.pkl','rb') as f:
            triang = pickle.load(f)
            
with open('smartwindows/Variables/point_sources_with_glass.pkl','rb') as f:
            point_sources = pickle.load(f)
            
with open('smartwindows/Variables/voltages_with_glass.pkl','rb') as f:
            V1, V2, V3, V4 = pickle.load(f)

trifinder = triang.get_trifinder()


y_b = 0.0e-5
y_t = 5.0e-5
t = 1.5e-5  
x_1 = 0.0
x_2 = 20e-5
b = 5e-5


tci = LinearTriInterpolator(triang,V1)
(Ex,Ey) = tci.gradient(triang.x,triang.y)
E = -np.array([Ex,Ey])


x = np.linspace(0,20e-5,200/4)
y = np.linspace(0-1.5e-5,5e-5+1.5e-5,(50+2*15)/4)
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


E_norm = np.sqrt(Ex**2 + Ey**2)


#plt.figure()
#plt.streamplot(X,Y,Ex1,Ey1,density=3)
#plt.title('Streamplot of Electric Field')
#plt.xlabel('x [m]')
#plt.ylabel('y [m]')
#plt.axis([0, 20e-5, -1.5e-5, 5e-5+1.5e-5])
#plt.ticklabel_format(style='sci', scilimits=(0,0)) 
#plt.show()

plt.figure()
c = plt.tricontourf(triang,V1)
plt.colorbar(c)
plt.title('Electric potential')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.ticklabel_format(style='sci', scilimits=(0,0)) 
plt.show()


# electric field
plt.figure()

plt.quiver(X,Y,Ex/E_norm,Ey/E_norm, color='blue')

plt.plot([x_1,x_2], [y_b, y_b],'black')
plt.plot([x_1,x_2], [y_t, y_t],'black')
plt.plot([x_1,x_2], [y_b-t, y_b-t],'black')
plt.plot([x_1,x_2], [y_t+t, y_t+t],'black')
plt.plot([x_1,x_1], [y_b-t, y_t+t],'black')
plt.plot([x_2,x_2], [y_b-t, y_t+t],'black')


plt.plot([x_1,x_1+b], [y_b,y_b], 'red')
plt.plot([x_1+2*b,x_1+3*b], [y_b,y_b], 'red')
plt.plot([x_1+b,x_1+2*b], [y_t,y_t], 'red')
plt.plot([x_1+3*b,x_2], [y_t,y_t], 'red')
plt.title('Electric field')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.ticklabel_format(style='sci', scilimits=(0,0)) 
plt.show()