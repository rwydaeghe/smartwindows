from smartwindow import *
from typing import Tuple, List, Callable, Union
from numbers import Number
from operator import attrgetter
from scipy.sparse import *
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, CubicTriInterpolator, LinearTriInterpolator
import random
import numpy as np
import traceback
import time
from tqdm import tqdm
import pickle

#constants
eps = np.finfo(float).eps 
e   = 1.602e-19
eps_0 = 8.854e-12
eps_r = 2

class Structure:
    def __init__(self,
                 shape: Tuple[float, float] = (20e-5, 5e-5),
                 grid_spacing: float = 1e-6):
        self.x, self.y = shape
        self.particles = []
        self.viscosity=1.344e-3
        self.density=750
        self.courant=0.7
        self.time_steps_passed=0
        self.t=0
        self.dt=5e-3
        
        self.electrode_cycle=200
        self.period=0
        self.electrode_config='initialising...'
        #dummyvariabele voor niet te blijven hangen aan de electrode
        self.voltage_up=False        
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
        if not self.contains(self.particles):
            self.keep_contained()
    
    def run(self, total_time: Number, animate: bool = True, electrode_values= [], **kwargs):
        total_time = self._int_time(total_time)
            
        time = range(0, total_time, 1)
        if electrode_values==[]:
            for times in range(total_time//self.electrode_cycle):
                electrode_values.append([[-50,0,0,0],[0,0,-50,0],[0,-50,0,0],[0,0,0,-50]])  
        plt.figure('Simulation')
        # plt.get_current_fig_manager().window.showMaximized()
        plt.ion()            
        for _ in tqdm(time):
            self.update_electrodes(electrode_values[self.period])
            self.update_E_particles(self.particles)
            self.update_forces(self.particles)
            self.update_particles(self.particles)
            if not self.contains(self.particles):
                self.keep_contained()
            self.time_steps_passed+=1
            if animate:
                self.visualize(**kwargs)
        if not animate:
            self.visualize(**kwargs)
        plt.ioff()     
        return np.linalg.norm(self.particles[0].forces['electrostatic'])

    def load_fields(self):
        print('loading fields')
        with open('Variables/voltages_small.pkl','rb') as f:
            self.V1, self.V2, self.V3, self.V4 = pickle.load(f)
        with open('Variables/triangulation_small.pkl','rb') as f:
            self.triang_V = pickle.load(f) 
            self.trifinder = self.triang_V.get_trifinder()
        with open('Variables/point_sources.pkl','rb') as f:
            self.point_sources = pickle.load(f) 

    def update_E_electrodes(self, x=[0,0,0,0]):
        self.V_electrodes = x[0]*self.V1 + x[1]*self.V2 + x[2]*self.V3 + x[3]*self.V4
        tci = LinearTriInterpolator(self.triang_V,self.V_electrodes) # faster interpolator, but not as accurate                             
        (Ex, Ey) = tci.gradient(self.triang_V.x,self.triang_V.y)
        self.E_electrodes = -np.array([Ex,Ey])
            
    def update_E_particles(self,particles,just_return_dont_update=False):
        self.V_point_sources=0
        for particle in particles:
            i = int(round(particle.pos[0]*10**6))
            j = int(round(particle.pos[1]*10**6))
            #to do: still a bug with particles being outside the period...
            #if not self.contains([particle]):
            #    print(self.particles_left.nnz,
            #          self.particles_right.nnz,
            #          self.particles_top.nnz,
            #          self.particles_bottom.nnz)
            self.V_point_sources += -self.point_sources[i,j]*particle.charge*e/(eps_0*eps_r)
        tci = LinearTriInterpolator(self.triang_V,self.V_point_sources) # faster interpolator, but not as accurate                             
        (Ex, Ey) = tci.gradient(self.triang_V.x,self.triang_V.y)        
        if just_return_dont_update:
            return -np.array([Ex,Ey])
        else:
            self.E_point_sources = -np.array([Ex,Ey])
        
    def update_electrodes(self,electrode_value):
        t=self.time_steps_passed
        t_c=self.electrode_cycle
        
        if t%t_c==t_c/4*0:
            self.electrode_config='bottom left'
            self.update_E_electrodes(x=electrode_value[0])
            self.voltage_up=False
            for particle in self.particles:
                particle.stagnant=False
        if t%t_c==t_c/4*1:
            self.electrode_config='top middle'
            self.update_E_electrodes(x=electrode_value[1])
            self.voltage_up=True
            for particle in self.particles:
                particle.stagnant=False
        if t%t_c==t_c/4*2:
            self.electrode_config='bottom middle'
            self.update_E_electrodes(x=electrode_value[2])
            self.voltage_up=False
            for particle in self.particles:
                particle.stagnant=False
        if t%t_c==t_c/4*3:
            self.electrode_config='top right'
            self.update_E_electrodes(x=electrode_value[3])
            self.voltage_up=True
            for particle in self.particles:
                particle.stagnant=False
        
        self.period=t//t_c   
        
    def update_forces(self, particles):
        cst=e**2/(4*np.pi*eps_0*eps_r)
        for particle in particles:
            particle.forces['coulomb']=np.array([0.0,0.0])
        for i,p1 in enumerate(particles):
            for _,p2 in enumerate(particles[i+1:]):
                r=p2.real_pos-p1.real_pos
                norm_r=np.linalg.norm(r)
                if norm_r<(p1.r+p2.r):
                    #collisions
                    self.collide_particles(p1, p2, r, norm_r)
                else:
                    #coulomb force
                    force=cst*p1.charge*p2.charge/norm_r**3*r
                    p1.forces['coulomb']+=-force
                    p2.forces['coulomb']+=force

            #electrostatic force
            p1.forces['electrostatic']=p1.charge*e*self.get_E(p1, p1.pos)
            
    def update_coulomb(self, from_particles, on_particle):
        cst=e**2/(4*np.pi*eps_0*eps_r)
        on_particle.forces['coulomb']=np.array([0.0,0.0])
        for from_particle in from_particles:
            if all(from_particle.pos==on_particle.pos):
                continue #we don't want self-interaction!
            r=from_particle.real_pos-on_particle.real_pos
            norm_r=np.linalg.norm(r)
            if norm_r<(on_particle.r+from_particle.r):
                #collisions
                self.collide_particles(on_particle, from_particle, r, norm_r)
            else:
                #coulomb force
                force=cst*on_particle.charge*from_particle.charge/norm_r**3*r
                on_particle.forces['coulomb']+=-force
                
    def update_electrostatic(self, particles):
        #electrostatic force
        for particle in particles:
            #if not particle.stagnant:
            particle.forces['electrostatic']=particle.charge*e*self.get_E(particle, particle.pos)
                
    def update_particles(self, particles):
        for particle in particles:
            #particle.update_force() #deprecated
            particle.update_pos()
            
    def get_field_here(self, field, pos: np.ndarray) -> np.ndarray:
        """could also be used for other fields like potentials"""
        tr = self.trifinder(*pos)                                  # triangle where particle is
        i = self.triang_V.triangles[tr]                                     # indices of vertices of tr
        v0 = np.array([self.triang_V.x[i[0]],self.triang_V.y[i[0]]])        # position of vertex 1
        v1 = np.array([self.triang_V.x[i[1]],self.triang_V.y[i[1]]])
        v2 = np.array([self.triang_V.x[i[2]],self.triang_V.y[i[2]]])
        norm = np.array([np.linalg.norm(v0-pos),np.linalg.norm(v1-pos),np.linalg.norm(v2-pos)])
        j = np.argmin(norm)                                                 # nearest vertex        
        v = i[j]
        fieldx = np.array(field[0])
        fieldy = np.array(field[1])
        return np.array([fieldx[v], fieldy[v]])
    
    @property
    def E(self):
        return self.E_electrodes+self.E_point_sources
    
    def get_E(self, particle=None, pos: np.ndarray=np.array([None,None])):
        """ 
        multiple purposes:
            -just the total field E if nothing is given
            -the total field on a position if that position is given
            -if also a particle is given, return E felt by that particle, i.e. without it's point-source field        
        """
        if pos[0]!=None:
            if particle!=None:
                return self.get_field_here(self.E-self.update_E_particles([particle],just_return_dont_update=True),pos)
            else:
                return self.get_field_here(self.E,pos)            
        else:
            return self.E

    def collide_particles(self, p1, p2, r12, norm_r):
        #elastic collision
        """
        v12=p2.vel-p1.vel
        val=2/(p1.m+p2.m)*np.dot(v12,r12)/norm_r**2*r12
        p1.vel+=p2.m*val
        p2.vel-=p1.m*val"""
        if p1.stagnant==True:
            p2.stagnant=True
            p2.pos=p1.pos+(p1.r+p2.r)*r12/norm_r                        
        if p2.stagnant==True:
            p1.stagnant=True
            p1.pos=p2.pos-(p1.r+p2.r)*r12/norm_r
        
    def keep_contained(self, particle_list=None, walls: str='all'):
        if particle_list==None:
            particle_list=self.particles
        for i, _ in zip(self.particles_left.col, self.particles_left.data):
            particle_list[i].collide(wall='left')
        for i, _ in zip(self.particles_right.col, self.particles_right.data):
            particle_list[i].collide(wall='right')
        for i, _ in zip(self.particles_bottom.col, self.particles_bottom.data):
            particle_list[i].collide(wall='bottom',voltage= not self.voltage_up)
        for i, _ in zip(self.particles_top.col, self.particles_top.data):
            particle_list[i].collide(wall='top',voltage=self.voltage_up)
                
                        
    def contains(self, particle_list: List) -> bool:
        """ Not only does this method return True if all particles are in 
        structure, it also updates lists of escaped particles by facet. Will 
        be called for all particles and maybe also individual particles """
        x,y,r = np.zeros(len(particle_list)), np.zeros(len(particle_list)), np.zeros(len(particle_list))
        for i, particle in enumerate(particle_list):
            x[i], y[i] = particle.pos
            r[i] = particle.r
        
        self.particles_left = coo_matrix(x < 0 + eps)
        self.particles_right = coo_matrix(x > self.x - eps)
        self.particles_bottom = coo_matrix(y - r < 0 + eps)
        self.particles_top = coo_matrix(y + r > self.y - eps)
         
        
        if any([self.particles_left.nnz,
                self.particles_right.nnz,
                self.particles_bottom.nnz,
                self.particles_top.nnz]):
            return False
        else:
            return True
        
    #for debugging
    def get_particles_attr(self, attr: str) -> np.ndarray:
        return np.array(list(map(attrgetter(attr), self.particles)))
    
    def _int_time(self, t: Union[int, float]) -> int:
        if isinstance(t, float):
            return int(t/self.dt)
        return t
    
    def visualize(self, with_field: str='nothing', **kwargs):
        plt.clf()        
        plt.xlabel('$x \/ \/ [m]$')
        plt.ylabel('$y \/ \/ [m]$')
        plt.xlim(0, self.x)
        plt.ylim(0, self.y)
        plt.title('Looking at period ' + str(self.period) + ' of structure. \n'
                  + 'Electrode configuration: only ' + self.electrode_config)
        plt.ticklabel_format(style='sci', scilimits=(0,0))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend(
            (plt.Circle((0,0), radius=5,color='b'),
             plt.Circle((0,0), radius=5,color='r'),
             plt.Circle((0,0), radius=5,color='g')
            ),
            ('Lagging behind period',
             'In current period',
             'Ahead of period',
             ), 
            loc=(0,-0.3))
            #bbox_to_anchor=(0,-0.8,1,0.2),
            #mode="expand",
            #ncol=3,
            #loc="lower center")
        #plt.tight_layout()
        
        if with_field!='nothing':
            x=np.linspace(0,self.x,200)
            y=np.linspace(0,self.y,50)
            X,Y=np.meshgrid(x,y)
            Ex=np.array([self.get_E(pos=np.array([x,y]))[0] for (x,y) in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
            Ey=np.array([self.get_E(pos=np.array([x,y]))[1] for (x,y) in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
            if with_field=='abs':
                Etot=np.array([np.linalg.norm(self.get_E(pos=np.array([x,y]))) for (x,y) in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
                plt.pcolor(X,Y,np.sqrt(Etot))
            elif with_field=='vector':
                plt.quiver(X,Y,Ex,Ey)
            elif with_field=='streamlines':
                plt.streamplot(X,Y,Ex,Ey,density=3)
        for particle in self.particles:
            particle.visualize(**kwargs)
            
        plt.pause(.01)
        plt.show()