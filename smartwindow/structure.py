from smartwindow import *
from typing import Tuple, List, Callable, Union
from numbers import Number
from operator import attrgetter
from scipy.sparse import *
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, CubicTriInterpolator, LinearTriInterpolator
import random
import numpy as np
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
        part_speed=5e-5
        self.courant=0.7
        self.time_steps_passed=0
        self.t=0
        self.dt=1e-1
        
        self.electrode_cycle=200
        self.period=0
        self.electrode_config='initialising...'
        
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
    
    def run(self, total_time: Number, animate: bool = True, visualizeWithField: bool=False):
        total_time = self._handle_time(total_time)
        time = range(0, total_time, 1)
            
        plt.figure('Simulation')
        plt.get_current_fig_manager().window.showMaximized()
        plt.ion()            
        for _ in tqdm(time):
            self.update_electrodes()           
            #self.add_point_sources(self.particles)
            self.update_forces(self.particles)
            self.update_particles(self.particles)
            if not self.contains(self.particles):
                self.keep_contained()
            self.time_steps_passed+=1
            if animate:
                self.visualize(visualizeWithField)
        if not animate:
            self.visualize(visualizeWithField)
        plt.ioff()      
        
    def load_fields(self):
        print('loading fields')
        with open('Variables/voltages_small.pkl','rb') as f:
            self.V1, self.V2, self.V3, self.V4 = pickle.load(f)
        with open('Variables/triangulation_small.pkl','rb') as f:
            self.triang_V = pickle.load(f) 
            self.trifinder = self.triang_V.get_trifinder()
        with open('Variables/point_sources.pkl','rb') as f:
            self.point_sources = pickle.load(f) 

    def update_fields(self, x1 : float = 0.0, x2 : float = 0.0, x3 : float = 0.0, x4 : float = 0.0):
        self.V = x1*self.V1 + x2*self.V2 + x3*self.V3 + x4*self.V4
        tci = LinearTriInterpolator(self.triang_V,-self.V) # faster interpolator, but not as accurate                             
        (Ex, Ey) = tci.gradient(self.triang_V.x,self.triang_V.y)
        self.E = np.array([Ex,Ey])            
        for particle in self.particles:
            particle.stagnant=False
            
    def add_point_sources(self,particles):
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
            self.V_point_sources += self.point_sources[i,j]*particle.charge*e/(eps_0*eps_r)
        tci = LinearTriInterpolator(self.triang_V,-self.V_point_sources) # faster interpolator, but not as accurate                             
        (Ex, Ey) = tci.gradient(self.triang_V.x,self.triang_V.y)
        self.E += np.array([Ex,Ey])
        
    def update_electrodes(self):
        t=self.time_steps_passed
        t_c=self.electrode_cycle
        
        if t%t_c==t_c/4*0:
            self.electrode_config='bottom left'
            self.update_fields(x1=1)
        if t%t_c==t_c/4*1:
            self.electrode_config='top middle'
            self.update_fields(x3=1)
        if t%t_c==t_c/4*2:
            self.electrode_config='bottom middle'
            self.update_fields(x2=1)
        if t%t_c==t_c/4*3:
            self.electrode_config='top right'
            self.update_fields(x4=1)
        
        self.period=t//t_c   
        
    def update_forces(self, particles):
        #coulomb force
        #"""
        cst=e**2/(4*np.pi*eps_0*eps_r)
        for i,p1 in enumerate(particles):
            for _,p2 in enumerate(particles[i+1:]):
                r=p2.real_pos-p1.real_pos
                norm_r=np.linalg.norm(r)
                if norm_r<(p1.r+p2.r):
                    self.collide_particles(p1, p2, r, norm_r)
                else:
                    force=cst*p1.charge*p2.charge/norm_r**3*r      
                    p1.forces['coulomb']=-force
                    p2.forces['coulomb']=force
        #"""
        #electrostatic force
        for particle in particles:
            force  = -particle.charge*e*self.get_electric_field(particle.pos)
            particle.forces['electrostatic']=force
        
        #check if fieldlines are normal to electrodes
        """
        reference_points=[np.array([self.x/4*(0+0.5),0]),
                          np.array([self.x/4*(1+0.5),self.y]),
                          np.array([self.x/4*(2+0.5),0]),
                          np.array([self.x/4*(3+0.5),self.y])]
        for refpoint in reference_points:
            Ex, Ey =self.get_electric_field(refpoint) 
            print(np.degrees(np.arctan(Ey/Ex))) #x-component should be zero
        """
        
        #print('coul', self.particles[0].forces['coulomb'])
        #print('elec', self.particles[0].forces['electrostatic'])
                
    def update_particles(self, particles):
        for particle in particles:
            #particle.update_force() #deprecated
            particle.update_pos()
            
    def get_electric_field(self, pos: np.ndarray) -> np.ndarray:
        tr = self.trifinder(*pos)                                  # triangle where particle is
        i = self.triang_V.triangles[tr]                                     # indices of vertices of tr
        v0 = np.array([self.triang_V.x[i[0]],self.triang_V.y[i[0]]])        # position of vertex 1
        v1 = np.array([self.triang_V.x[i[1]],self.triang_V.y[i[1]]])
        v2 = np.array([self.triang_V.x[i[2]],self.triang_V.y[i[2]]])
        norm = np.array([np.linalg.norm(v0-pos),np.linalg.norm(v1-pos),np.linalg.norm(v2-pos)])
        j = np.argmin(norm)                                                 # nearest vertex        
        v = i[j]
        Ex = np.array(self.E[0])
        Ey = np.array(self.E[1])
        return np.array([Ex[v], Ey[v]])

    def collide_particles(self, p1, p2, r12, norm_r):
        #elastic collision
        """
        v12=p2.vel-p1.vel
        val=2/(p1.m+p2.m)*np.dot(v12,r12)/norm_r**2*r12
        p1.vel+=p2.m*val
        p2.vel-=p1.m*val"""
        if p1.stagnant==True:
            p2.pos=p1.pos+(p1.r+p2.r)*r12/norm_r
            p2.stagnant=True            
        if p2.stagnant==True:
            p1.stagnant=True
            p1.pos=p2.pos-(p1.r+p2.r)*r12/norm_r
        
    def keep_contained(self, walls: str='all'):
        if walls=='top_and_bottom':
            for i, _ in zip(self.particles_bottom.col, self.particles_bottom.data):
                self.particles[i].collide(wall='bottom')
            for i, _ in zip(self.particles_top.col, self.particles_top.data):
                self.particles[i].collide(wall='top')
        elif walls=='all':
            for i, _ in zip(self.particles_left.col, self.particles_left.data):
                self.particles[i].collide(wall='left')
            for i, _ in zip(self.particles_right.col, self.particles_right.data):
                self.particles[i].collide(wall='right')
            for i, _ in zip(self.particles_bottom.col, self.particles_bottom.data):
                self.particles[i].collide(wall='bottom')
            for i, _ in zip(self.particles_top.col, self.particles_top.data):
                self.particles[i].collide(wall='top')
                        
    def contains(self, particle_list: List) -> bool:
        """ Not only does this method return True if all particles are in 
        structure, it also updates lists of escaped particles by facet. Will 
        be called for all particles and maybe also individual particles """
        t=time.clock()
        x,y,r = np.zeros(len(particle_list)), np.zeros(len(particle_list)), np.zeros(len(particle_list))
        for i, particle in enumerate(particle_list):
            x[i], y[i] = particle.pos
            r[i] = particle.r
        
        self.particles_left = coo_matrix(x < 0 + eps)
        self.particles_right = coo_matrix(x > self.x - eps)
        self.particles_bottom = coo_matrix(y - r < 0 + eps)
        self.particles_top = coo_matrix(y + r > self.y - eps)
        
        if (self.particles_left.nnz != 0 &
            self.particles_right.nnz != 0 &
            self.particles_bottom.nnz != 0 &
            self.particles_top.nnz != 0):
            return True
        else:
            return False
        
    #for debugging
    def get_particles_attr(self, attr: str) -> np.ndarray:
        return np.array(list(map(attrgetter(attr), self.particles)))
    
    def _handle_time(self, t: Union[int, float]) -> int:
        if isinstance(t, float):
            return int(t/self.dt)
        return t
    
    def visualize(self, withField: bool=False):
        plt.clf()
        for particle in self.particles:
            particle.visualize()
        plt.xlabel('$x \/ \/ [m]$')
        plt.ylabel('$y \/ \/ [m]$')
        plt.xlim(0, self.x)
        plt.ylim(0, self.y)
        plt.title('Looking at period ' + str(self.period) + ' of structure. \n'
                  + 'Electrode configuration: only ' + self.electrode_config)
        plt.ticklabel_format(style='sci', scilimits=(0,0))
        plt.gca().set_aspect('equal', adjustable='box')
        
        if withField:
            x=np.linspace(0,self.x,100)
            y=np.linspace(0,self.y,100)
            X,Y=np.meshgrid(x,y)
            #Ex=np.array([self.get_electric_field(np.array([x,y]))[0] for (x,y) in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
            #Ey=np.array([self.get_electric_field(np.array([x,y]))[1] for (x,y) in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
            #plt.quiver(X,Y,Ex,Ey)
            Etot=np.array([np.linalg.norm(self.get_electric_field(np.array([x,y]))) for (x,y) in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
            plt.pcolor(X,Y,np.sqrt(Etot))
        
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
        plt.pause(.1)
        plt.show()
            

        