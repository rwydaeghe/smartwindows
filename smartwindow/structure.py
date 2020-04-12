from smartwindow import *
from typing import Tuple, List, Callable
from numbers import Number
from operator import attrgetter
from scipy.sparse import *
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, CubicTriInterpolator, LinearTriInterpolator
import random
import numpy as np
import time
import pickle

#floating point machine precision
eps = np.finfo(float).eps 

class Structure:
    def __init__(self,
                 shape: Tuple[float, float] = (20e-5, 5e-5),
                 grid_spacing: float = 1e-6):
        self.grid_spacing = float(grid_spacing)
        self.Nx, self.Ny = self._handle_tuple(shape)
        self.time_steps_passed = 0
        self.particles = []
        self.viscosity=1.344e-3
        self.density=750
        part_speed=5e-5
        self.courant=0.7
#        self.dt = self.courant*self.grid_spacing/part_speed
        self.dt=1e-2
        
    def add_particle(self, particle):
        particle._register_structure(self)
        self.particles.append(particle)
    
    def add_gaussian_particle_cloud(
        self, 
        N: int = 100,
        avg_pos: Tuple[float, float] = None,
        var_pos: Tuple[float, float] = None,
        avg_charge: int = 100,
        var_charge: int = 40,
        avg_size: float = 2.5e-7,
        var_size: float = 1e-7
    ):
        if avg_pos == None:
            avg_pos = np.array(self.size)/2
        else:
            avg_pos = np.array(avg_pos)
        if var_pos == None:
            var_pos = np.array(self.size)/10
        else:
            var_pos = np.array(var_pos)
        
        positions = list(zip(np.random.normal(avg_pos[0],var_pos[0], N),np.random.normal(avg_pos[1],var_pos[1], N)))
        charges = np.random.normal(avg_charge,var_charge, N)
        sizes = np.random.normal(avg_size, var_size, N)
        
        for i in range(N):
            self.add_particle(
                 Particle(pos = positions[i], charge = charges[i], r = sizes[i])
                 )
    
    def run(self, total_time: Number, animate: bool = True):
        if isinstance(total_time, float):
            total_time /= self.dt
        time = range(0, int(total_time), 1)
            
        plt.figure('Simulation')
        plt.ion()            
        for t in time:
            #if t%200==0:
            #    self.update_fields(x1=2)
            #self.update_forces()
            self.update_particles()
            if not self.contains(self.particles):
                self.keep_contained()
            if animate:
                self.visualize()
                plt.pause(0.01)
        if not animate:
            self.visualize()
        #plt.ioff()            
    
    def update_fields(self, x1 : float = 1.0, x2 : float = 0.0, x3 : float = 0.0, x4 : float = 0.0):
        with open('Variables/voltages.pkl','rb') as f:
            V1, V2, V3, V4 = pickle.load(f)
        with open('Variables/triangulation.pkl','rb') as f:
            triang_V = pickle.load(f)    
        V = x1*V1 + x2*V2 + x3*V3 + x4*V4
        tci = LinearTriInterpolator(triang_V,-V)                                # faster interpolator, but not as accurate                             
#        tci = CubicTriInterpolator(triang_V, -V)                              
        (Ex, Ey) = tci.gradient(triang_V.x,triang_V.y)
        self.E = np.array([Ex,Ey])
        trifinder = triang_V.get_trifinder()
#        with open('Variables/electric_field.pkl','rb') as f:                    # eventueel werken met opgeslagen velden
#            self.E = pickle.load(f)
        
        print(self.E)
        
        print("update fields")
        for particle in self.particles:
            tr = trifinder(particle.pos[0], particle.pos[1])                    # triangle where particle is
            i = triang_V.triangles[tr]                                          # indices of vertices
            v0 = np.array([triang_V.x[i[0]],triang_V.y[i[0]]])                  # position of vertex 1
            v1 = np.array([triang_V.x[i[1]],triang_V.y[i[1]]])
            v2 = np.array([triang_V.x[i[2]],triang_V.y[i[2]]])
            norm = np.array([np.linalg.norm(v0),np.linalg.norm(v1),np.linalg.norm(v2)])
            j = np.argmin(norm)                                                 # nearest vertex        
            v = i[j]
            Ex = np.array(self.E[0])
            Ey = np.array(self.E[1])
            E = np.array([Ex[v], Ey[v]])
            force  = particle.charge*1.602e-19*E
            particle.forces['electrostatic']=force
    
        
    def update_forces(self):
        #coulomb force
        for i,p1 in enumerate(self.particles):
            for _,p2 in enumerate(self.particles[i+1:]):
                r=p2.pos-p1.pos
                vec_r=r/np.linalg.norm(r)
                force=2.307e-28*p1.charge*p2.charge/r**2*vec_r
                #force*=2
                p1.forces["coulomb"]=force
                p2.forces["coulomb"]=-force
                
    def update_particles(self):
        for particle in self.particles:
            particle.update()
        #print(self.particles[0].forces, self.particles[0].force)
        self.time_steps_passed += 1
    
    def keep_contained(self):
        for i, _ in zip(self.particles_left.col, self.particles_left.data):
            self.particles[i].collide(wall='left')
            #self.particles_left[i]=False
        for i, _ in zip(self.particles_right.col, self.particles_right.data):
            self.particles[i].collide(wall='right')
            #self.particles_right[i]=False
        for i, _ in zip(self.particles_bottom.col, self.particles_bottom.data):
            self.particles[i].collide(wall='bottom')
            #self.particles_bottom[i]=False
        for i, _ in zip(self.particles_top.col, self.particles_top.data):
            self.particles[i].collide(wall='top')
            #self.particles_top[i]=False
            
    def contains(self, particle_list: List):
        """ Not only does this method return True if all particles are in 
        structure, it also updates lists of escaped particles by facet. Will 
        be called for all particles and maybe also individual particles """
        t=time.clock()
        x,y,r = np.zeros(len(particle_list)), np.zeros(len(particle_list)), np.zeros(len(particle_list))
        for i, particle in enumerate(particle_list):
            x[i], y[i] = particle.pos
            r[i] = particle.r
        
        
        self.particles_left = coo_matrix(x - r < 0 + eps)
        self.particles_right = coo_matrix(x + r > self.x - eps)
        self.particles_bottom = coo_matrix(y - r < 0 + eps)
        self.particles_top = coo_matrix(y + r > self.y - eps)
        
        if (self.particles_left.nnz != 0 &
            self.particles_right.nnz != 0 &
            self.particles_bottom.nnz != 0 &
            self.particles_top.nnz != 0):
            return True
    
    def _handle_distance(self, distance: Number) -> int:
        """ transform a distance to an integer number of gridpoints """
        if not isinstance(distance, int):
            return int(float(distance) / self.grid_spacing + 0.5)
        return distance    
        
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
        
    #for debugging
    def get_particles_attr(self, attr: Tuple) -> np.ndarray:
        return np.array(list(map(attrgetter(*attr), self.particles)))
    
    @property
    def x(self) -> int:
        return self.Nx * self.grid_spacing

    @property
    def y(self) -> int:
        return self.Ny * self.grid_spacing

    @property
    def size(self) -> Tuple[int, int]:
        """ get the size of the FDTD grid, in meters """
        return (self.x, self.y)
    
    @property
    def shape(self) -> Tuple[int, int]:
        """ get the shape of the FDTD grid, in cells """
        return (self.Nx, self.Ny)
            
    @property
    def time_passed(self) -> float:
        """ get the total time passed """
        return self.time_steps_passed * self.dt
    
    def visualize(self):
        plt.clf()
        for particle in self.particles:
            particle.visualize()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(0, self.x)
        plt.ylim(0, self.y)
        plt.ticklabel_format(style='sci', scilimits=(0,0))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.pause(.1)
        plt.show()
            

        