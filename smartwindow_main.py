from smartwindow import *
import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.close('all')

print('initializing')
structure = Structure((20e-5, 5e-5))

print('adding particles')
#structure.add_gaussian_particle_cloud(N=50, avg_pos=(1e-5,2.5e-5), var_pos=(1e-5,1e-5))        
#structure.add_gaussian_particle_cloud(N=50, avg_pos=(10e-5,2.5e-5), var_pos=(1e-5,1e-5), avg_charge=-100)

structure.add_gaussian_particle_cloud(N=50, avg_pos=(10e-5,2.5e-5), var_pos=(1e-5,1e-5))

#structure.add_particle(Particle(pos = (10e-5-1e-5,2.25e-5), charge = 50, r = 5e-7))
#structure.add_particle(Particle(pos = (11e-5,3.5e-5), charge = 50, r = 5e-7,c='b'))
#structure.add_particle(Particle(pos = (11.5e-5,3.5e-5), charge = 50, r = 5e-7,c='b'))
#structure.add_particle(Particle(pos = (11.5e-5,3.5e-5), vel=(-1e-5,-1e-5), charge = 50, r = 5e-7,c='g'))
#structure.add_particle(Particle(pos = (1e-5,0.5e-5), vel=(-1e-5,-1e-5), charge = 50, r = 5e-7,c='g'))
periods = []
structure.load_fields()
structure.run(200,with_field='nothing',with_arrows=False)
for particle in structure.particles:
    periods.append(particle.structure_period)
result= np.bincount(np.array(periods)+1)
plt.close('all')

#simulation=Simulation(structure)
#simulation.run(600*1e-1)

