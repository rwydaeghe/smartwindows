from typing import Tuple, List, Callable
from numbers import Number
import matplotlib.pyplot as plt
import numpy as np

class Particle:
    def __init__(self,
                 pos: Tuple[float,float] = (0.0,0.0),
                 vel: Tuple[float,float] = (0.0,0.0),
                 acc: Tuple[float,float] = (0.0,0.0),
                 charge: int = 100,
                 r: float = 3e-7,
                 m: float = 3e-9,
                 c: str = 'r'
        ):
        self.structure = None
        
        self.pos = np.array(pos)
        self.structure_period=0
        self.vel = np.array(vel)
        self.vel_col = np.array([0.0,0.0])
        self.acc = np.array(acc)
        self.charge=charge
        self.r=r
        self.density=4230
        self.volume=4/3*np.pi*self.r**3
        self.m=self.density*self.volume
        self.color=c
        
        self.stagnant=False
        
        self.force = self.m*self.acc
        self.forces = {'electrostatic': np.array([0.0,0.0]), 
                       'coulomb':       np.array([0.0,0.0]), 
                       'stokes':        np.array([0.0,0.0])}
        
    def _register_structure(self, structure):
        self.structure = structure
        self.stokes_coeff = 6*np.pi*self.structure.viscosity*self.r
#        print(self.m/self.stokes_coeff)
        
    def update_force(self):
        self.force=np.array([0.0,0.0])
        self.forces['stokes'] = -self.stokes_coeff*self.vel
        for force_contribution in self.forces.values():
            self.force+=force_contribution
        
    def update_pos(self):
        #self.acc=self.force/self.m
        #self.vel+=self.acc*self.structure.dt
        
        if self.stagnant:
            self.vel=np.array([0.0,0.0])
        else:            
            self.vel=(self.forces['electrostatic']+self.forces['coulomb'])/self.stokes_coeff
        #self.vel+=self.vel_col
        self.pos+=self.vel*self.structure.dt
        
        if self.structure_period==self.structure.period:
            self.color='r'
        elif self.structure_period<self.structure.period:
            self.color='b'
        elif self.structure_period>self.structure.period:
            self.color='g'
        
    def collide(self, wall: str):
        if wall=='left':
            self.pos[0]=self.structure.x-self.r*1.1 #perio RVW
            self.structure_period -= 1
        if wall=='right':
            self.pos[0]=self.r*1.1 #perio RVW
            self.structure_period += 1
        if wall=='bottom':
            self.pos[1]=self.r*1.1            
            #self.vel[1]*=-1 #elastisch
            self.stagnant=True
        if wall=='top':
            self.pos[1]=self.structure.y-self.r*1.1
            #self.vel[1]*=-1 #elastisch
            self.stagnant=True
            
    @property
    def real_pos(self):
        return np.array([self.pos[0]+self.structure.x*self.structure_period,self.pos[1]])
        
    def visualize(self):
        #self.disp_pos=self.pos
        #self.disp_pos[0]=self.disp_pos[0]%self.structure.x
        #plt.gca().add_artist(plt.Circle(self.disp_pos, self.r, color=self.color))
        
        plt.gca().add_artist(plt.Circle(self.pos, self.r, color=self.color))