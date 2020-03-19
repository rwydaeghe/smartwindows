# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 19:42:46 2020

@author: rbnwy
"""
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

#f = lambda x,y,z: (.1*(1+.5*np.sin(2*np.pi*x/10)),.1,.1)
fx = lambda x: .1*(1+.5*np.heaviside(x-3.333,1)-.5*np.heaviside(x-6.667,1))
fy = lambda y: .1
fz = lambda z: .1
diagonal_spacings = [[],[],[]]
diagonal_positions = [np.array([0]),np.array([0]),np.array([0])]

while diagonal_positions[0][-1] < 10:
    diagonal_spacings[0].append(fx(diagonal_positions[0][-1]))
    diagonal_positions[0]=np.append(diagonal_positions[0],
                          diagonal_positions[0][-1]+diagonal_spacings[0][-1])
while diagonal_positions[1][-1] < 10:
    diagonal_spacings[1].append(fy(diagonal_positions[1][-1]))
    diagonal_positions[1]=np.append(diagonal_positions[1],
                          diagonal_positions[1][-1]+diagonal_spacings[1][-1])
while diagonal_positions[2][-1] < 10:
    diagonal_spacings[2].append(fz(diagonal_positions[2][-1]))
    diagonal_positions[2]=np.append(diagonal_positions[2],
                          diagonal_positions[2][-1]+diagonal_spacings[2][-1])
diagonal_positions = np.array(diagonal_positions)
diagonal_spacings = np.array(diagonal_spacings)
print(diagonal_positions, diagonal_spacings)

#plt.scatter(diagonal_positions[:,1],diagonal_positions[:,0])
#plt.scatter(diagonal_positions[:,1],np.append(diagonal_spacings[:,0],diagonal_spacings[-1,0]))
#plt.show()