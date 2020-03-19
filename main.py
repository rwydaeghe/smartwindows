""" FDTD Example

A simple example on how to use the FDTD Library

"""

## Imports

import fdtd
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

## Constants
WAVELENGTH = 5e-6
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light

# create FDTD Grid
grid = fdtd.Grid(
    (2.5e-5, 1.5e-5),
    spacing_functions=(lambda x: .1*WAVELENGTH*(1-.5*np.heaviside(x-1.0e-5,1)+.5*np.heaviside(x-1.5e-5,1)),
                       lambda y: .1*WAVELENGTH),
    permittivity=1.0,
    permeability=1.0,
    conductivity=0.0,
    courant_number=0.6
)

grid[8e-6:12e-6, 5e-6:10e-6] = fdtd.Object(permittivity=10, permeability=1, conductivity=0, name="object")

grid[15e-6, 5e-6:10e-6] = fdtd.LineSource(period=WAVELENGTH / SPEED_LIGHT, power=10, name="source")


grid.run(100, animate=True)

