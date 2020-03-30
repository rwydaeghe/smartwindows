""" FDTD Objects

The objects to place in the grid. Objects define all the regions in the grid
with a modified update equation, such as for example regions with anisotropic
permittivity etc.

Available Objects:
 - Object
 - AnisotropicObject

"""

## Imports

from typing import Union, List
ListOrSlice = Union[List[int], slice]

# relative
from .grid import Grid
import numpy as np

## Object
class Object:
    """ An object to place in the grid """

    def __init__(self, permittivity: float, permeability: float, name: str = None):
        """ Create an object """
        self.grid = None
        self.name = name
        self.permittivity = permittivity
        self.permeability = permeability

    def _register_grid(
        self, grid: Grid, x: ListOrSlice, y: ListOrSlice
    ):
        """ Register the object to the grid

        Args:
            grid: the grid to register the object into
            x: the x-location of the object in the grid
            y: the y-location of the object in the grid
        """
        self.grid = grid
        self.grid.objects.append(self)
        if self.name is not None:
            if not hasattr(grid, self.name):
                setattr(grid, self.name, self)
            else:
                raise ValueError(
                    f"The grid already has an attribute with name {self.name}"
                )
        self.x = self._handle_slice(x, max_index=self.grid.Nx)
        self.y = self._handle_slice(y, max_index=self.grid.Ny)

        self.Nx = abs(self.x.stop - self.x.start)
        self.Ny = abs(self.y.stop - self.y.start)

        self.grid._set_material_properties((self.permittivity, self.permeability), positions=(x,y))        
        # set the permittivity values of the object at its border to be equal
        # to the grid permittivity. This way, the object is made symmetric.
        # no idea what this is. so delete for clutter
        # ik snap waarom: eps_x aan linkerkant van object is eps_object maar aan rechterkant is eps_x,y,z allemaal van grid. dat y en z van grid zijn is oké maar eps_x aan rechterrand toch best wijzen IN het object (het is niet alsof eps_x een vectoriele afhankelijkheid heeft hé)
        
        #self.grid.permittivity[self.x, self.y] = np.inf        
    def _handle_slice(self, s: ListOrSlice, max_index: int = None) -> slice:
        if isinstance(s, list):
            if len(s) == 1:
                return slice(s[0], s[0] + 1, None)
            raise IndexError(
                "One can only use slices or single indices to index the grid for an Object"
            )
        if isinstance(s, slice):
            start, stop, step = s.start, s.stop, s.step
            if step is not None and step != 1:
                raise IndexError(
                    "Can only use slices with unit step to index the grid for an Object"
                )
            if start is None:
                start = 0
            if stop is None:
                stop = max_index
            return slice(start, stop, None)
        raise ValueError("Invalid grid indexing used for object")   

    def update_E(self, curl_H):
        """ custom update equations for inside the object

        Args:
            curl_H: the curl of magnetic field in the grid.

        """
        loc = (self.x, self.y)
        self.grid.E[loc] += (
            self.grid.courant_number / self.permittivity * curl_H[loc]
        )

    def update_H(self, curl_E):
        """ custom update equations for inside the object

        Args:
            curl_E: the curl of electric field in the grid.

        """

    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)})"

    def __str__(self):
        s = "    " + repr(self) + "\n"

        def _handle_slice(s):
            return (
                str(s)
                .replace("slice(", "")
                .replace(")", "")
                .replace(", ", ":")
                .replace("None", "")
            )

        x = _handle_slice(self.x)
        y = _handle_slice(self.y)
        s += f"        @ x={x}, y={y}".replace(":,", ",")
        if s[-1] == ":":
            s = s[:-1]
        return s + "\n"