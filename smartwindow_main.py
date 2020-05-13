from smartwindow import *
import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.close('all')

print('initializing')
structure = Structure((20e-5, 5e-5))

print('adding particles')
#structure.add_gaussian_particle_cloud(N=50, avg_pos=(1e-5,2.5e-5), var_pos=(1e-5,1e-5))        
#structure.add_gaussian_particle_cloud(N=50, avg_pos=(10e-5,2.5e-5), var_pos=(1e-5,1e-5))

#structure.add_gaussian_particle_cloud(N=50, avg_pos=(10e-5,2.5e-5), var_pos=(1e-5,1e-5))

structure.add_particle(Particle(pos = (10e-5-1e-5,2.25e-5), charge = 50, r = 5e-7))
structure.add_particle(Particle(pos = (11e-5,3.5e-5), charge = 50, r = 5e-7))
structure.add_particle(Particle(pos = (11.5e-5,3.5e-5), charge = 50, r = 5e-7))

#structure.add_particle(Particle(pos = (11.5e-5,3.5e-5), vel=(-1e-5,-1e-5), charge = 50, r = 5e-7,c='g'))
#structure.add_particle(Particle(pos = (1e-5,0.5e-5), vel=(-1e-5,-1e-5), charge = 50, r = 5e-7,c='g'))
values=[]
for i in range(2000//structure.electrode_cycle):
    values.append([[-50,0,0,0],[0,0,-50,0],[0,-50,0,0],[0,0,0,-50]])
periods = []
structure.load_fields()
structure.run(2000,with_field='nothing',with_arrows=False,electrode_values=values)
for particle in structure.particles:
    periods.append(particle.structure_period)
result= np.bincount(np.array(periods)+1)
plt.close('all')

simulation=Simulation(structure)
simulation.run(40,animate=True,with_field='nothing',with_arrows=True)


""" only for debugging coulomb and point-source forces
y=np.zeros(9)
for i in range(1,10):
    print(i)
    structure = Structure((20e-5, 5e-5))
    structure.add_particle(Particle(pos = (10e-5-i*5e-6,2.5e-5), charge = 50, r = 5e-7))
    structure.add_particle(Particle(pos = (10e-5+i*5e-6,2.5e-5), charge = 50, r = 5e-7))
    structure.load_fields()
    y[i-1]=structure.run(1,with_field='vector',with_arrows=True)
plt.figure('t')
x=np.array(list(range(1,10)))*2*5e-6
#plt.loglog(x,np.sqrt(y)*x)
plt.loglog(x,y)
plt.show()
"""
