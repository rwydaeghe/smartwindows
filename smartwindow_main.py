from smartwindow import *
import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.close('all')

print('initializing')
structure = Structure((20e-5, 5e-5))

print('adding particles')
structure.add_gaussian_particle_cloud(N=50, avg_pos=(1e-5,2.5e-5), var_pos=(1e-5,1e-5))        
#structure.add_gaussian_particle_cloud(N=50, avg_pos=(18e-5,2.5e-5), var_pos=(1e-5,1e-5))       

#structure.add_gaussian_particle_cloud(N=500, avg_pos=(10e-5,2.5e-5), var_pos=(10e-5,2e-5))         

#structure.add_particle(Particle(pos = (10e-5-1e-5,2.25e-5), vel=(1e-5,0.25e-5), charge = 50, r = 5e-7))
#structure.add_particle(Particle(pos = (11e-5,3.5e-5), vel=(-1e-5,-1e-5), charge = 50, r = 5e-7,c='b'))
#structure.add_particle(Particle(pos = (14e-5,3.5e-5), vel=(-1e-5,-1e-5), charge = 50, r = 5e-7,c='g'))

structure.load_fields()

structure.run(600)
