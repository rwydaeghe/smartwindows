from smartwindow import *
import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.close('all')

print('initializing')
structure = Structure((20e-5, 5e-5))

print('adding particles')
structure.add_gaussian_particle_cloud(N=50, avg_pos=(1e-5,2.5e-5), var_pos=(3e-5,3e-5))        
#structure.add_particle(Particle(pos = (1e-5,2.5e-5), charge = 50, r = 2.5e-7))

structure.load_fields()

structure.run(300)
