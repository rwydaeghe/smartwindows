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
        self.t=0        
        
        #All the various time-scales ordered: order must always (!) be respected when inputting new values
        self.dx_imposed=1e-6 #indirectly related to the smallest time-scale
        self.dt={'on_electrode_change': 5e-4,
                 'initial':             self.structure.dt,
                 'visual':              5e-3,
                 'long_range':          1e-2,
                 'fast_forward':        2e-1,
                 'electrode_cycle':     5e0} #one electrode switch is a 1/4th of this
        self.steps = dict.fromkeys(self.dt,0)
        self.electrodes_just_changed=False
        
    def run(self, run_time, animate: bool = True, fast_forward=True, **kwargs):
        plt.figure('Simulation')
        plt.get_current_fig_manager().window.showMaximized()
        plt.ion()
        run_time=self._int_time(run_time)
        
        self.start_time=self.t
        while self.t<self.start_time+run_time:
            #print the time
            sys.stdout.write("\rt="+str(self.t))
            sys.stdout.flush()
                        
            #long-range time-scale:
            if self.t_passes_dt('long_range'):
                #this refers to the largest time-scale. most of the time doesn't do anything:
                self.update_electrodes()                
                self.structure.update_E_particles(self.particles)
                self.structure.update_electrostatic(self.particles)
                if self.steps['long_range']==1:
                    self.initialize_events()
                if self.electrodes_just_changed:
                    self.reset_stagnant_particles()
                    self.electrodes_just_changed=False            
                     
            if fast_forward & (self.events.size==0):
                """if all particles are stagnant, skip the iteration and fast-foward some time
                good when particles are really quick and low amounts
                reduces accuracy, especially if overshooting future events with to big dt['fast_forward']"""
                #to do: potentially unnecessary as we'll wake them in reset_stagnant_particles
                for particle in self.particles:
                    particle.t=self.t
                continue 
            
            #smallest time-scale:
            self.youngest_event=self.events[:,0]
            self.youngest_particle=self.get_particle(0)            
            self.do_event(self.youngest_event)
            self.t=self.youngest_particle.t
            self.youngest_particle.dt*=self.dx_imposed/self.youngest_particle.last_dx
            self.structure.update_coulomb(self.particles, self.youngest_particle) #to do: only update close encounters!
            if not self.structure.contains([self.youngest_particle]):
                self.structure.keep_contained([self.youngest_particle])
                """the following is a terrible way of coding but it works:"""
                if not self.youngest_particle.stagnant:
                    """... so it must have periodically moved"""
                    self.advance_event(0,periodic=True)
                else:
                    self.events=np.delete(self.events,0,1)
            else:
                if not self.youngest_particle.stagnant:                    
                    self.advance_event(0)
                else:
                    """... so it must have gotten infected by a collision in update_coulomb"""
                    self.events=np.delete(self.events,0,1)                
            
            if self.t_passes_dt('visual') & animate: #all time-scales below the visual one get visually uniformified (but not e.g. fast-forward)
                self.structure.visualize(**kwargs)
 
        if not animate:
            self.structure.visualize(**kwargs)
        plt.ioff()
            
    def do_event(self, event):
        particle=self.particles[int(event[0])]
        particle.set_space_time(event)
        
    def advance_event(self, event_index, periodic: bool=False):
        if periodic:
            self.events[1:3,event_index]=self.get_particle(event_index).pos
            self.events[3,event_index]=self.t #immediately do it
        else:
            self.events[1:,event_index]=self.get_particle(event_index).next_space_time
        self.events=self.events[:,self.events[3,:].argsort(kind='stable')] #efficiently sorts the almost perfectly sorted events by time
        
    def initialize_events(self):
        self.events=np.zeros((4,len(self.particles)))
        for i,particle in enumerate(self.particles):
            particle.dt=self.dt['initial']
            self.events[0,i]=i
            self.events[1:,i]=particle.next_space_time
        self.events=self.events[:,self.events[3,:].argsort()] #sorts events by time            
    
    def reset_stagnant_particles(self):
        for i,particle in enumerate(self.particles):
            if particle.stagnant:
                particle.stagnant=False
                particle.t=self.t #if you wake someone up from a coma, you have to tell him what year it is
                particle.dt=self.dt['on_electrode_change']
                """Note that you can perfectly just get rid of above line 
                and it'll take the last dt ever updated the particle with.
                Generally that dt should quite small upon entry of electrode,
                so that's good, but especially for gaussian clouds, not good
                enough. dt['on_electrode_change'] may/should be absurdily low
                """
                newevent=np.zeros((4,1))
                newevent[0,0]=i
                newevent[1:,0]=particle.next_space_time
                self.events=np.hstack((self.events,newevent))
        
    def update_electrodes(self):
        #to do when we'll be doing more complex switching phenomena (each their own dt):
        #use the t_passes_dt for each timescale of electrode switch
        if ((self.t%self.dt['electrode_cycle']>=self.dt['electrode_cycle']/4*0)
          & (self.t%self.dt['electrode_cycle']<=self.dt['electrode_cycle']/4*1)
         & ((self.structure.electrode_config=='top right')
          | (self.structure.electrode_config=='initialising...'))):
            self.structure.electrode_config='bottom left'
            self.structure.update_E_electrodes(x=[-100,0,0,0])
            self.structure.voltage_up=False
            self.electrodes_just_changed=True
        elif ((self.t%self.dt['electrode_cycle']>self.dt['electrode_cycle']/4*1)
            & (self.t%self.dt['electrode_cycle']<=self.dt['electrode_cycle']/4*2)
            & (self.structure.electrode_config=='bottom left')):
            self.structure.electrode_config='top middle'
            self.structure.update_E_electrodes(x=[0,0,-100,0])
            self.structure.voltage_up=True
            self.electrodes_just_changed=True
        elif ((self.t%self.dt['electrode_cycle']>self.dt['electrode_cycle']/4*2)
            & (self.t%self.dt['electrode_cycle']<=self.dt['electrode_cycle']/4*3)
            & (self.structure.electrode_config=='top middle')):
            self.structure.electrode_config='bottom middle'
            self.structure.update_E_electrodes(x=[0,-100,0,0])
            self.structure.voltage_up=False
            self.electrodes_just_changed=True
        elif ((self.t%self.dt['electrode_cycle']>self.dt['electrode_cycle']/4*3)
            & (self.t%self.dt['electrode_cycle']<=self.dt['electrode_cycle']/4*4)
            & (self.structure.electrode_config=='bottom middle')):
            self.structure.electrode_config='top right'
            self.structure.update_E_electrodes(x=[0,0,0,-100])
            self.structure.voltage_up=True
            self.electrodes_just_changed=True
        
        self.t_passes_dt('electrode_cycle')
        #note that steps is which step is currently being processed, not the amount of steps done
        self.structure.period=self.steps['electrode_cycle']-1
        
    def get_particle(self, event_index: int):
        particle_id=int(self.events[0,event_index])
        return self.particles[particle_id]
    
    def t_passes_dt(self, timescale: str):
        """ Will return true once if t passes a dt mark. 
        Subsequent calling in that iteration returns false. 
        Also updates to correct step being processed. """
        if (self.t//self.dt[timescale]>=self.steps[timescale]):
            self.steps[timescale]=int(self.t//self.dt[timescale]+1)             
            return True
        else:
            #print(self.t,self.dt[timescale],self.t//self.dt[timescale],self.steps[timescale])
            #raise ValueError("dt['"+timescale+ "'] too short")
            return False
        
    def _int_time(self, t: Union[int, float]) -> int:
        if isinstance(t, float):
            return int(t/self.dt)
        return t
        