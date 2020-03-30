from typing import Tuple, List, Callable
from numbers import Number
import matplotlib.pyplot as plt
import numpy as np

class Particle:
    def __init__(self,
                 pos: Tuple[float,float] = (0.0,0.0),
                 vel: Tuple[float,float] = (2e-4,0.0),
                 acc: Tuple[float,float] = (0.0,0.0),
                 charge: int = 100,
                 r: float = 3e-7,
                 m: float = 3e-9
        ):
        self.structure = None
        
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self.acc = np.array(acc)
        self.charge=charge
        self.r=r
        self.m=self.r/100
        self.color='r'
        
        self.force = self.m*self.acc
        self.forces = {'electrostatic': np.array([0.0,0.0]), 'coulomb': np.array([0.0,0.0]), 'stokes': np.array([0.0,0.0]), 'collision': np.array([0.0,0.0])}
        
    def _register_structure(self, structure):
        self.structure = structure
        self.stokes_coeff = 6*np.pi*self.structure.viscosity*self.r        
        
    def update_force(self):
        self.force=np.array([0.0,0.0])
        self.forces['stokes'] = -self.stokes_coeff*self.vel
        for force_contribution in self.forces.values():
            self.force+=force_contribution
        
    def update(self):
        self.update_force()
        
        self.acc=self.force/self.m
        self.vel+=self.acc*self.structure.dt
        self.pos+=self.vel*self.structure.dt
    
    def collide(self, wall: str):
        if wall=='left':
            self.pos[0]=self.structure.x-self.r*1.1
        if wall=='right':
            self.pos[0]=self.r*1.1
        if wall=='bottom':
            self.pos[1]=self.r*1.1
            #elastisch
            self.vel[1]*=-1
        if wall=='top':
            print('t')
            self.pos[1]=self.structure.y-self.r*1.1
            #elastisch
            self.vel[1]*=-1
        
    def visualize(self):
        plt.gca().add_artist(plt.Circle(self.pos, self.r, color=self.color))