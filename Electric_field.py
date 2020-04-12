# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:41:58 2020

@author: Hanne
"""

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, CubicTriInterpolator
import matplotlib.cm as cm
import numpy as np
import pickle


with open('smartwindows/Variables/voltages.pkl','rb') as f:
    V1, V2, V3, V4 = pickle.load(f)
with open('smartwindows/Variables/triangulation.pkl','rb') as f:
    triang_V = pickle.load(f)    
x = triang_V.x
y = triang_V.y
tci1 = CubicTriInterpolator(triang_V, -V1)
tci2 = CubicTriInterpolator(triang_V, -V2)
tci3 = CubicTriInterpolator(triang_V, -V3)
tci4 = CubicTriInterpolator(triang_V, -V4)
(Ex1, Ey1) = tci1.gradient(x,y)
(Ex2, Ey2) = tci2.gradient(x,y)
(Ex3, Ey3) = tci3.gradient(x,y)
(Ex4, Ey4) = tci4.gradient(x,y)


E1 = [Ex1,Ey1]
E2 = [Ex2,Ey2]
E3 = [Ex3,Ey3]
E4 = [Ex4,Ey4]

#Plot triangulation, potential and vector field
fig, ax = plt.subplots()
ax.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax.use_sticky_edges = False
ax.margins(0.07)

#ax.triplot(triang_V, color='0.8')

#levels = np.arange(0., 1., 0.01)
cmap = cm.get_cmap(name='hot', lut=None)
ax.tricontour(x, y, V, cmap=cmap)
# Plots direction of the electrical vector field
ax.quiver(x_n, y_n, Ex_n/E_n, Ey_n/E_n,units='xy',
          width=0.007e-5, headwidth=3e-5, headlength=4e-5)

ax.set_title('Gradient plot')
plt.show()