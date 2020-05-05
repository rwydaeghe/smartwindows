from smartwindow import *
from typing import Tuple, List, Callable, Union
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

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
            particle.dt=particle.charge/100*1e-1
            self.events[0,i]=i
            self.events[1:,i]=particle.next_space_time
        
        self.events=self.events[:,self.events[3,:].argsort()] #sorts events by time
        
        
        while self.t<run_time:
            self.structure.update_electrodes()
            self.structure.add_point_sources(self.particles)
            self.structure.update_forces(self.particles)
            
            sys.stdout.write("\rt="+str(self.t))
            sys.stdout.flush()

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
        self.t+=event[3]        
        
    def advance_event(self, event_index):
        particle_id=int(self.events[0,event_index])
        particle=self.particles[particle_id]
        
        self.events[1:,event_index]=particle.next_space_time
        self.events=self.events[:,self.events[3,:].argsort(kind='stable')] #efficiently sorts the almost perfectly sorted events by time
       
    @property
    def youngest_event(self):
        return self.events[:,0]