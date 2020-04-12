""" FDTD Example

A simple example on how to use the FDTD Library

"""

## Imports

from smartwindow import *
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

print('initializing')
structure = Structure((20e-5, 5e-5))

print('adding particles')
structure.add_gaussian_particle_cloud(N=50)        
#self.add_particle(Particle(pos = np.array(self.size)*0.5, charge = 50, r = 5e-7))
#self.add_particle(Particle(pos = np.array(self.size)*0.4, charge = 50, r = 5e-7))

#print(structure.get_particles_attr('r'))

print('start fields')
structure.update_fields(x1=2)

structure.run(100)
