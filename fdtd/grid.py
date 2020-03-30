""" The FDTD Grid

The grid is the core of the FDTD Library. It is where everything comes
together and where the biggest part of the calculations are done.

"""

## Imports

# 3rd party
from tqdm import tqdm
from typing import Tuple, List, Callable
from numbers import Number

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

## Constants
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light


## Functions
def curl_E(E: np.ndarray) -> np.ndarray:
    """ Transforms an E-type field into an H-type field by performing a curl
    operation

    Args: E: Electric field to take the curl of (E-type field located on
        integer grid points)

    Returns: curl_E: the curl of E (H-type field located on half-integer grid
        points)
    """
    curl = np.zeros(E.shape)
    
    """ 
    '1:' stelt i+1 voor, 
    ':-1' stelt i voor (da's ook waarom die altijd in LL staat)
    """

    curl[:, :-1, 0] += E[:, 1:, 2] - E[:, :-1, 2]

    curl[:-1, :, 1] -= E[1:, :, 2] - E[:-1, :, 2]

    curl[:-1, :, 2] += E[1:, :, 1] - E[:-1, :, 1]
    curl[:, :-1, 2] -= E[:, 1:, 0] - E[:, :-1, 0]

    return curl

def curl_H(H: np.ndarray) -> np.ndarray:
    """ Transforms an H-type field into an E-type field by performing a curl
    operation

    Args:
        H: Magnetic field to take the curl of (H-type field located on
            half-integer grid points)

    Returns:
        curl_H: the curl of H (E-type field located on integer grid points)
    """
    curl = np.zeros(H.shape)
    
    """ 
    '1:' stelt i+1 voor, 
    ':-1' stelt i voor (da's ook waarom die altijd in LL staat)
    """
    
    """
    curl[:, :-1, 0] += H[:, 1:, 2] - H[:, :-1, 2]

    curl[:-1, :, 1] -= H[1:, :, 2] - H[:-1, :, 2]

    curl[:-1, :, 2] += H[1:, :, 1] - H[:-1, :, 1]
    curl[:, :-1, 2] -= H[:, 1:, 0] - H[:, :-1, 0]
    """
    
    curl[:, 1:, 0] += H[:, 1:, 2] - H[:, :-1, 2]

    curl[1:, :, 1] -= H[1:, :, 2] - H[:-1, :, 2]

    curl[1:, :, 2] += H[1:, :, 1] - H[:-1, :, 1]
    curl[:, 1:, 2] -= H[:, 1:, 0] - H[:, :-1, 0]

    return curl

## FDTD Grid Class
class Grid:
    """ The FDTD Grid

    The grid is the core of the FDTD Library. It is where everything comes
    together and where the biggest part of the calculations are done.

    """

    from .visualization import visualize

    def __init__(
        self,
        shape: Tuple[float, float],
        grid_spacing: float = 1e-6,
        permittivity: float = 1.0,
        permeability: float = 1.0,
        courant_number: float = None,
    ):
        """
        Args:
            shape: shape of the FDTD grid.
            grid_spacing = 50e-9: distance between the grid cells.
            permittivity = 1.0: the relative permittivity of the background.
            permeability = 1.0: the relative permeability of the background.
            courant_number = None: the courant number of the FDTD simulation.
                Defaults to the inverse of the square root of the number of
                dimensions > 1 (optimal value). The timestep of the simulation
                will be derived from this number using the CFL-condition.
        """
        
        self.grid_spacing = float(grid_spacing)
        self.Nx, self.Ny = self._handle_tuple(shape)
        self.V, self.F = self.meshRectangle() #similar to CSWP
        
        self.K = self.assemble()
        
        self.BC=np.zeros((self.V.shape[0],))
        self.BC[0:26]=np.ones((26,))*self.grid_spacing**2
        self.BC[390:]=2*np.ones((26,))*self.grid_spacing**2
        
        print(self.K.shape,spsolve(self.K,self.BC))
        # courant number of the simulation (optimal value)
        if courant_number is None:
            # slight stability factor added
            self.courant_number = 0.99 * float(2**(-0.5))
        elif courant_number > float(2**(-0.5)):
            raise ValueError(
                f"courant_number {courant_number} too high"
            )
        else:
            self.courant_number = float(courant_number)
        
        self.dt = self.courant_number * self.grid_spacing / SPEED_LIGHT
        
        # save electric and magnetic field
        self.E = np.zeros((self.Nx, self.Ny, 3))
        self.H = np.zeros((self.Nx, self.Ny, 3))

        # save the material properties
        self.permittivity = np.zeros((self.Nx, self.Ny, 3))
        self.permeability = np.zeros((self.Nx, self.Ny, 3))
        
        # set material properties
        self._set_material_properties((permittivity, permeability))
        
        # save current time index
        self.time_steps_passed = 0

        # dictionary containing the sources:
        self.sources = []

        # dictionary containing the boundaries
        self.boundaries = []

        # dictionary containing the detectors
        self.detectors = []

        # dictionary containing the objects in the grid
        self.objects = []

    def _handle_distance(self, distance: Number) -> int:
        """ transform a distance to an integer number of gridpoints """
        if not isinstance(distance, int):
            return int(float(distance) / self.grid_spacing + 0.5)
        return distance

    def _handle_time(self, time: Number) -> int:
        """ transform a time value to an integer number of timesteps """
        if not isinstance(time, int):
            return int(round(float(time) / self.dt + 0.5))
        return time

    def _handle_tuple(
        self, shape: Tuple[Number, Number]
    ) -> Tuple[int, int]:
        """ validate the grid shape and transform to a length-2 tuple of ints """
        if len(shape) != 2:
            raise ValueError(
                f"invalid grid shape {shape}\n"
                f"grid shape should be a 2D tuple containing floats or ints"
            )
        x, y = shape
        x = self._handle_distance(x)
        y = self._handle_distance(y)
        return x, y

    def _handle_slice(self, s: slice) -> slice:
        """ validate the slice and transform possibly float values to ints """
        start = (
            s.start if not isinstance(s.start, float) else self._handle_distance(s.start)
        )
        stop = (
            s.stop if not isinstance(s.stop, float) else self._handle_distance(s.stop)
        )
        step = (
            s.step if not isinstance(s.step, float) else self._handle_distance(s.step)
        )
        return slice(start, stop, step)

    def _handle_single_key(self, key):
        """ transform a single index key to a slice or list """
        try:
            len(key)
            return [self._handle_distance(k) for k in key]
        except TypeError:
            if isinstance(key, slice):
                return self._handle_slice(key)
            else:
                return [self._handle_distance(key)]
        return key
        
    @property
    def x(self) -> int:
        return self.Nx * self.grid_spacing

    @property
    def y(self) -> int:
        return self.Ny * self.grid_spacing

    @property
    def shape(self) -> Tuple[int, int]:
        """ get the shape of the FDTD grid """
        return (self.Nx, self.Ny)

    @property
    def time_passed(self) -> float:
        """ get the total time passed """
        return self.time_steps_passed * self.dt

    def cartesian_product(self, *arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[...,i] = a
        return arr

    def meshRectangle(self):
        #create vertices
        x = np.linspace(0,self.x,self.Nx+1)
        y = np.linspace(0,self.y,self.Ny+1)
        X,Y=np.meshgrid(x,y)
        V=np.transpose(np.vstack([X.ravel(), Y.ravel()]))
        
        #create faces based on index of V. Left to right, then down to up
        x_counter=np.linspace(0,self.Nx-1,self.Nx)
        xy_counter=(self.Nx+1)*np.linspace(0,self.Ny-1,self.Ny)
        face_counter=np.array([0,1,self.Nx+2,self.Nx+1]) #anti clock-wise
        bottom_left=(x_counter + np.transpose(np.array([xy_counter,]*self.Nx))).ravel()
        F=(face_counter + np.transpose(np.array([bottom_left,]*face_counter.size))).astype(int)
        
        return V, F
    
    def assemble(self):
        #local stiffness matrix
        e=self.grid_spacing**2
        k=np.zeros((16,self.F.shape[0]))
        k[[0,5,10,15],:]=4*e
        k[[1,3,6,11],:]=-e
        k[[2,7],:]=-2*e
        k[[4,8,9,12,13,14],:]=k[[1,2,6,3,7,11],:]
        k=np.transpose(k)
        
        #assemble
        I=self.F[:,[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]]
        J=self.F[:,[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]]
        K=sparse.coo_matrix((k.ravel(),(I.ravel(),J.ravel())),shape=(self.V.shape[0],self.V.shape[0]))
        
        return K.tocsr()
        
    def run(self, total_time: Number, progress_bar: bool = False, animate: bool = False):
        """ run an FDTD simulation.

        Args:
            total_time: the total time for the simulation to run.
            progress_bar = True: choose to show a progress bar during
                simulation

        """
        if isinstance(total_time, float):
            total_time /= self.dt
        time = range(0, int(total_time), 1)
        if progress_bar:
            time = tqdm(time)
            
        plt.figure()
        plt.ion()            
        for _ in time:
            self.step()
            if animate:
                self.visualize()
                plt.pause(0.01)
        if not animate:
            self.visualize()
        plt.ioff()            

    def step(self):
        """ do a single FDTD step by first updating the electric field and then
        updating the magnetic field
        """
        self.update_E()
        self.update_H()
        self.time_steps_passed += 1

    def update_E(self):
        """ update the electric field by using the curl of the magnetic field """

        # update boundaries: step 1
        for boundary in self.boundaries:
            boundary.update_phi_E()

        curl = curl_H(self.H)
        self.E += self.courant_number / self.permittivity * curl
        
        # update objects
        #for obj in self.objects:
            #obj.update_E(curl)

        # update boundaries: step 2
        for boundary in self.boundaries:
            boundary.update_E()

        # add sources to grid:
        for src in self.sources:
            src.update_E()

        # detect electric field
        for det in self.detectors:
            det.detect_E()

    def update_H(self):
        """ update the magnetic field by using the curl of the electric field """

        # update boundaries: step 1
        for boundary in self.boundaries:
            boundary.update_phi_H()

        curl = curl_E(self.E)
        self.H -= self.courant_number / self.permeability * curl

        # update objects
        for obj in self.objects:
            obj.update_H(curl)

        # update boundaries: step 2
        for boundary in self.boundaries:
            boundary.update_H()

        # add sources to grid:
        for src in self.sources:
            src.update_H()

        # detect electric field
        for det in self.detectors:
            det.detect_H()

    def reset(self):
        """ reset the grid by setting all fields to zero """
        self.H *= 0.0
        self.E *= 0.0
        self.time_steps_passed *= 0

    def add_source(self, name, source):
        """ add a source to the grid """
        source._register_grid(self)
        self.sources[name] = source

    def add_boundary(self, name, boundary):
        """ add a boundary to the grid """
        boundary._register_grid(self)
        self.boundaries[name] = boundary

    def add_detector(self, name, detector):
        """ add a detector to the grid """
        detector._register_grid(self)
        self.detectors[name] = detector

    def add_object(self, name, obj):
        """ add an object to the grid """
        obj._register_grid(self)
        self.objects[name] = obj
        
    def _set_material_properties(self, materialproperties: tuple = (1.0,1.0,0.0), positions: tuple = None):
        permittivity, permeability = materialproperties
          
        if positions != None:
            x, y = positions
            Nx, Ny = abs(x.stop - x.start), abs(y.stop - y.start)
        else:
            x, y = slice(None), slice(None)
            Nx, Ny = self.Nx, self.Ny
                  
        # save the yee-fdtd coefficients (not defined for the last points)
        ones=np.ones((Nx, Ny, 3))        
        self.permittivity[x, y] = ones * float(permittivity)
        self.permeability[x, y] = ones * float(permeability)
            
    def __setitem__(self, key, attr):
        if not isinstance(key, tuple):
            x, y = key, slice(None)
        elif len(key) == 1:
            x, y = key[0], slice(None)
        elif len(key) == 2:
            x, y = key
        else:
            raise KeyError("maximum number of indices for the grid is 2")
        
        attr._register_grid(
            grid=self,
            x=self._handle_single_key(x),
            y=self._handle_single_key(y)            
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(shape=({self.Nx},{self.Ny}), "
            f"courant_number={self.courant_number:.2f})"
        )

    def __str__(self):
        """ string representation of the grid

        lists all the components and their locations in the grid.
        """
        s = repr(self) + "\n"
        if self.sources:
            s = s + "\nsources:\n"
            for src in self.sources:
                s += str(src)
        if self.detectors:
            s = s + "\ndetectors:\n"
            for det in self.detectors:
                s += str(det)
        if self.boundaries:
            s = s + "\nboundaries:\n"
            for bnd in self.boundaries:
                s += str(bnd)
        if self.objects:
            s = s + "\nobjects:\n"
            for obj in self.objects:
                s += str(obj)
        return s
