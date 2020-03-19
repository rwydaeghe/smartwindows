""" FDTD Sources

Sources are objects that inject power into the grid. Available Sources:

- LineSource

"""
## Imports

# other
from math import pi, sin

# typing
from typing import Tuple, Union, List
from numbers import Number
ListOrSlice = Union[List[int], slice]

import time
# relatvie
from .grid import Grid
import numpy as np

## LineSource class
class LineSource:
    """ A source along a line in the FDTD grid """

    def __init__(
        self,
        period: Number = 15,
        power: float = 1.0,
        polarization: int = 2,
        phase_shift: float = 0.0,
        name: str = None,
    ):
        """ Create a LineSource with a gaussian profile

        Args:
            period = 1: The period of the source. The period can be specified
                as integer [timesteps] or as float [seconds]
            power = 1.0: The power of the source
            phase_shift = 0.0: The phase offset of the source.

        """
        self.grid = None
        self.period = period
        self.power = power
        self.polarization = polarization        
        self.phase_shift = phase_shift
        self.name = name

    def _register_grid(
        self, grid: Grid, x: ListOrSlice, y: ListOrSlice
    ):
        """ Register a grid for the source.

        Args:
            grid: the grid to place the source into.
            x: The x-location of the source in the grid
            y: The y-location of the source in the grid

        Note:
            As its name suggests, this source is a LINE source.
            Hence the source spans the diagonal of the cube
            defined by the slices in the grid.
        """
        self.grid = grid
        self.grid.sources.append(self)
        if self.name is not None:
            if not hasattr(grid, self.name):
                setattr(grid, self.name, self)
            else:
                raise ValueError(
                    f"The grid already has an attribute with name {self.name}"
                )

        self.x, self.y = self._handle_slices(x, y)

        self.period = grid._handle_time(self.period)
        amplitude = (
            self.power / self.grid.permittivity[self.x, self.y, self.polarization]
        ) ** 0.5
        
        L = len(self.x)
        self.vect = np.array(
            (np.array(self.x) - self.x[L // 2]) ** 2
            + (np.array(self.y) - self.y[L // 2]) ** 2,
            np.float,
        )

        self.profile = np.exp(-self.vect ** 2 / (2 * (0.5 * self.vect.max()) ** 2))
        self.profile /= self.profile.sum()
        self.profile *= amplitude

    def _handle_slices(
        self, x: ListOrSlice, y: ListOrSlice
    ) -> Tuple[List, List]:
        """ Convert slices in the grid to lists

        This is necessary to make the source span the diagonal of the volume
        defined by the slices.

        Args:
            x: The x-location of the volume in the grid
            y: The y-location of the volume in the grid

        Returns:
            x, y: the x and y coordinates of the source as lists

        """

        # if list-indices were chosen:
        if isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                raise IndexError(
                    "sources require grid to be indexed with slices or equal length list-indices"
                )
            return x, y

        # if a combination of list-indices and slices were chosen,
        # convert the list-indices to slices.
        # TODO: maybe issue a warning here?
        if isinstance(x, list):
            x = slice(x[0], x[-1], None)
        if isinstance(y, list):
            y = slice(y[0], y[-1], None)

        # if we get here, we can assume slices:
        x0 = x.start if x.start is not None else 0
        y0 = y.start if y.start is not None else 0
        x1 = x.stop if x.stop is not None else self.grid.Nx
        y1 = y.stop if y.stop is not None else self.grid.Ny

        # we can now convert these coordinates into index lists
        m = max(abs(x1 - x0), abs(y1 - y0))
        x = [v.item() for v in np.array(np.linspace(x0, x1, m, endpoint=False), np.int)]
        y = [v.item() for v in np.array(np.linspace(y0, y1, m, endpoint=False), np.int)]
        
        return x, y

    def update_E(self):
        """ Add the source to the electric field """
        q = self.grid.time_steps_passed
        vect = self.profile * sin(2 * pi * q / self.period + self.phase_shift)
        # do not use list indexing here, as this is much slower especially for torch backend
        # DISABLED: self.grid.E[self.x, self.y, 2] = self.vect
        # TO DO IS HET OOK ECHT TRAGER
        #t=time.time()
        for x, y, value in zip(self.x, self.y, vect):
            self.grid.E[x, y, self.polarization] += value
        #print(time.time()-t)

    def update_H(self):
        """ Add the source to the magnetic field """
        #only if magnetic sources were to exist (which they don't)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(period={self.period}, "
            f"power={self.power}, phase_shift={self.phase_shift}, "
            f"name={repr(self.name)})"
        )

    def __str__(self):
        s = "    " + repr(self) + "\n"
        x = f"[{self.x[0]}, ... , {self.x[-1]}]"
        y = f"[{self.y[0]}, ... , {self.y[-1]}]"
        s += f"        @ x={x}, y={y}\n"
        return s
