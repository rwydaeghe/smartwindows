""" FDTD Example

A simple example on how to use the FDTD Library

"""

## Imports

from smartwindow import *
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

structure = Structure((16e-5, 5e-5))

#print(structure.is_in(structure.get_particles_attr(pos),structure.get_particles_attr(r)))
#print(structure.get_particles_attr('r'))

structure.run(100)
