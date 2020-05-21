from smartwindow import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

plt.close('all')

print('initializing')
structure = Structure((20e-5, 5e-5))

print('adding particles')
structure.add_gaussian_particle_cloud(N=100, avg_pos=(10e-5,2.5e-5), var_pos=(1e-5,1e-5), seed=0)
#structure.add_gaussian_particle_cloud(N=50, avg_pos=(10e-5,2.5e-5), var_pos=(1e-5,1e-5))

#structure.add_gaussian_particle_cloud(N=50, avg_pos=(10e-5,2.5e-5), var_pos=(1e-5,1e-5))

#structure.add_particle(Particle(pos = (10e-5,2.5e-5), charge = 50, r = 5e-7))
#structure.add_particle(Particle(pos = (11e-5,2e-5), charge = 50, r = 5e-7))
#structure.add_particle(Particle(pos = (12e-5,3e-5), charge = 50, r = 5e-7))

#structure.add_particle(Particle(pos = (11.5e-5,3.5e-5), vel=(-1e-5,-1e-5), charge = 50, r = 5e-7,c='g'))
#structure.add_particle(Particle(pos = (1e-5,0.5e-5), vel=(-1e-5,-1e-5), charge = 50, r = 5e-7,c='g'))
#"""
simulation=Simulation(structure)
values=[]
for i in range(2000//structure.electrode_cycle):
    values.append([[-50,0,0,0],[0,0,-50,0],[0,-50,0,0],[0,0,0,-50]])
periods = []
structure.load_fields()
print('starting simulation')
#"""
structure.run(5.0,
              animate=True,
              with_field='nothing',
              with_arrows=False,
              electrode_values=values)
#"""
"""
simulation.run(0.5,
               animate=True,
               animate_events=False,
               with_field='nothing',
               with_arrows=False,
               electrode_values=values)
"""
print(structure.get_particles_attr('structure_period'))
for particle in structure.particles:
    periods.append(particle.structure_period)
result= np.bincount(np.array(periods)+1)
#plt.close('all')

