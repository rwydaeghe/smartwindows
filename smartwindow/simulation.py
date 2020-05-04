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
k_e = 8.987e9
e   = 1.602e-19
eps_0 = 8.854e-12
eps_r = 2

class Simulation:
    def __init__(self,structure):
        self.structure=structure
        self.particles=self.structure.particles
        self.smallest_dt=self.structure.dt
        self.t=0
        self.dt_visual=1e-1
    def run(self, run_time, animate: bool = True, visualizeWithField: bool=False):
        plt.figure('Simulation')
        plt.get_current_fig_manager().window.showMaximized()
        plt.ion()
        self.events=np.zeros((4,len(self.particles)))
        for i,particle in enumerate(self.particles):
            particle.dt=particle.charge/100
            self.events[0,i]=i
            self.events[1:,i]=particle.next_space_time
        
        self.events=self.events[:,self.events[3,:].argsort()] #sorts events by time
        
        self.structure.update_electrodes()
        self.structure.add_point_sources(self.particles)
        self.structure.update_forces(self.particles)
        self.structure.update_particles(self.particles)
        
        while self.t<run_time:
            self.do_event(self.youngest_event)
            self.advance_event(0)
            
            if not self.structure.contains(self.particles):
                self.structure.keep_contained()
            if animate:
                self.structure.visualize(visualizeWithField)
        if not animate:
            self.structure.visualize(visualizeWithField)
        plt.ioff()
        
    def do_event(self, event):
        particle=self.particles[int(event[0])]
        particle.set_space_time(event)
        
    def advance_event(self, event_number):
        particle_id=int(self.events[0,event_number])
        particle=self.particles[particle_id]
        
        print(self.events)
        self.events[1:,event_number]=particle.next_space_time
        print(self.events)        
        self.events=self.events[:,self.events[3,:].argsort(kind='stable')] #efficiently sorts the almost perfectly sorted events by time
        print(self.events)
        """
        time_index=np.searchsorted(self.events[3,1:],
                                   self.events[3,event_number])
        print(self.events)
        print(time_index)
        self.events=np.insert(
                    self.events[:,1:],
                    time_index,
                    self.events[:,event_number],
                    axis=1)
        print(self.events)
        """
        plt.pause(2)
       
    @property
    def youngest_event(self):
        return self.events[:,0]