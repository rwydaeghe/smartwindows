from smartwindow import *
from typing import Tuple, List, Callable, Union
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
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
        self.dx_max=5e-7 #indirectly related to the smallest time-scale. Twice the average radii of the particles
        self.dt={'on_electrode_change': 5e-3,
                 'initial':             5e-3, #important: will decide time-scale of "slow" particles and has significant influence
                 'visual':              5e-3, #visually uniformifies everything below it
                 'long_range':          1e-2,
                 'fast_forward':        2e-1,
                 'electrode_cycle':     5e0} #one electrode switch is a 1/4th of this
        self.steps = dict.fromkeys(self.dt,0)
        
        self.P3M_sphere_radius=3e-6/2 #1e-6 is the size of a cell in Fenic's field calculation
        
        self.electrodes_just_changed=False
        
    def run(self, 
        run_time, 
        animate: bool = True,
        animate_events: bool = False,        
        fast_forward=True, 
        electrode_values= [], 
        **kwargs
    ):
        plt.figure('Simulation')
        #plt.get_current_fig_manager().window.showMaximized()
        plt.ion()
        run_time=self._float_time(run_time)
        
        if electrode_values==[]:
            for times in range(total_time//self.electrode_cycle):
                electrode_values.append([[-100,0,0,0],[0,0,-100,0],[0,-100,0,0],[0,0,0,-100]])
        
        self.start_time=self.t        
        while self.t<self.start_time+run_time:
            #print the time
            sys.stdout.write("\rt="+str(self.t))
            sys.stdout.flush()

            #long-range time-scale:
            if self.t_passes_dt('long_range'):
                #this refers to the largest time-scale. most of the time doesn't do anything:
                self.update_electrodes(electrode_values[self.structure.period])                
                
                #the actual long_range time-scale:
                self.structure.apply_electrostatic(self.particles)
                self.update_P3M_spheres(self.particles,self.P3M_sphere_radius) #To do: give only list of non-stagnant particles
                """
                self.structure.update_E_PM(self.particles)                
                for particle in self.particles:
                    if not particle.stagnant: #this 'if' gives a big performance boost
                        self.structure.apply_PM(exempted_particles=self.get_neighbours(particle),
                                                on_particle=particle)
                """
                #initialization
                if self.steps['long_range']==1:
                    for particle in self.particles:
                        self.structure.apply_PP(self.particles, particle)
                    self.initialize_events()
                
                #see first line
                if self.electrodes_just_changed:
                    self.reset_stagnant_particles()
                    self.electrodes_just_changed=False       
                     
            if fast_forward & (self.events.size==0):
                """if all particles are stagnant, skip the iteration and fast-foward some time
                good when particles are really quick and low amounts
                reduces accuracy, especially if overshooting future events with to big dt['fast_forward']"""
                self.t+=self.dt['fast_forward']
                #to do: potentially unnecessary as we'll wake them in reset_stagnant_particles                
                for particle in self.particles:
                    particle.t=self.t
                continue 
            
            #smallest time-scale:
            self.youngest_event=self.events[:,0]
            self.youngest_particle=self.get_particle(0)
            self.do_event(self.youngest_event)
            self.t=self.youngest_particle.t
            self.set_adaptive_dt(self.youngest_particle)            
            self.structure.apply_PP(self.get_neighbours(self.youngest_particle),
                                    self.youngest_particle)
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
                    """... so it must have gotten infected by a collision in apply_PP"""
                    self.events=np.delete(self.events,0,1)
            
            #animation
            if animate_events:
                """mainly for educational and debugging purposes"""
                animate=False
                plt.figure('Space time')
                self.visualize()
                plt.pause(0.01)
                plt.figure('Simulation')
            if self.t_passes_dt('visual') & animate: 
                """all time-scales below the visual one get visually uniformified (but not e.g. fast-forward)"""
                self.structure.visualize(**kwargs)
                plt.pause(0.01)
 
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
        
    def set_adaptive_dt(self, particle):
        #note that "dx" is really just dr
        #for future: you can be very creative in how you choose this time step
        if self.youngest_particle.last_dx > self.dx_max:
            particle.dt*=self.dx_max/particle.last_dx
        
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
                Generally that dt should be quite small upon entry of electrode,
                so that's good, but especially for gaussian clouds, not good
                enough. dt['on_electrode_change'] may/should be absurdily low
                """
                newevent=np.zeros((4,1))
                newevent[0,0]=i
                newevent[1:,0]=particle.next_space_time
                self.events=np.hstack((self.events,newevent))
                
    def update_P3M_spheres(self,particles,r):
        #Based on future events:
        """
        points=np.transpose(self.events[1:3,:])
        self.tree = spatial.cKDTree(points)
        particle_event_idx=self.tree.query_ball_tree(self.tree, r)
        print(self.get_particle_id(particle_event_idx))
        """
        #Based on current positions:
        points=np.zeros((len(particles),2))
        for i,particle in enumerate(particles):
            #if not particle.stagnant: #TO DO
            points[i,:]=particle.pos
        self.tree = spatial.cKDTree(points)
        self.neighbours=self.tree.query_ball_tree(self.tree, r) #in terms of particle_id's
    
    def get_neighbours(self, particle):        
        return [self.particles[neighbour_id] for neighbour_id in self.neighbours[particle.id]]        
        
    def update_electrodes(self,electrode_value):
        #to do when we'll be doing more complex switching phenomena (each their own dt):
        #use the t_passes_dt for each timescale of electrode switch
        if ((self.t%self.dt['electrode_cycle']>=self.dt['electrode_cycle']/4*0)
          & (self.t%self.dt['electrode_cycle']<=self.dt['electrode_cycle']/4*1)
         & ((self.structure.electrode_config=='top right')
          | (self.structure.electrode_config=='initialising...'))):
            self.structure.electrode_config='bottom left'
            self.structure.update_E_electrodes(x=electrode_value[0])
            self.structure.voltage_up=False
            self.electrodes_just_changed=True
        elif ((self.t%self.dt['electrode_cycle']>self.dt['electrode_cycle']/4*1)
            & (self.t%self.dt['electrode_cycle']<=self.dt['electrode_cycle']/4*2)
            & (self.structure.electrode_config=='bottom left')):
            self.structure.electrode_config='top middle'
            self.structure.update_E_electrodes(x=electrode_value[1])
            self.structure.voltage_up=True
            self.electrodes_just_changed=True
        elif ((self.t%self.dt['electrode_cycle']>self.dt['electrode_cycle']/4*2)
            & (self.t%self.dt['electrode_cycle']<=self.dt['electrode_cycle']/4*3)
            & (self.structure.electrode_config=='top middle')):
            self.structure.electrode_config='bottom middle'
            self.structure.update_E_electrodes(x=electrode_value[2])
            self.structure.voltage_up=False
            self.electrodes_just_changed=True
        elif ((self.t%self.dt['electrode_cycle']>self.dt['electrode_cycle']/4*3)
            & (self.t%self.dt['electrode_cycle']<=self.dt['electrode_cycle']/4*4)
            & (self.structure.electrode_config=='bottom middle')):
            self.structure.electrode_config='top right'
            self.structure.update_E_electrodes(x=electrode_value[3])
            self.structure.voltage_up=True
            self.electrodes_just_changed=True
        
        self.t_passes_dt('electrode_cycle')
        #note that steps is which step is currently being processed, not the amount of steps done
        self.structure.period=self.steps['electrode_cycle']-1
        
    def get_particle_id(self, event_index: int):
        return int(self.events[0,event_index])
    
    def get_particle(self, event_index: int):
        return self.particles[self.get_particle_id(event_index)]
    
    def visualize(self):
        plt.clf()        
        plt.xlabel('$x \/ \/ [m]$')
        plt.ylabel('$t \/ \/ [m]$')
        plt.xlim(0, self.structure.x)
        plt.title('Events in space-time for x-direction')
        plt.ticklabel_format(style='sci', scilimits=(0,0))
        
        xpos=self.events[1,:]
        times=self.events[3,:]
        plt.ylim((self.t//(self.dt['electrode_cycle']/4))*(self.dt['electrode_cycle']/4),max(times))
        plt.scatter(xpos,times,s=3,c='r')
        for i,event in enumerate(self.events.T):
            particle=self.get_particle(i)
            plt.scatter(particle.pos[0],particle.t,s=3,c='b')
            neighbours=self.get_neighbours(particle)
            for neighbour in neighbours:
                plt.plot([particle.pos[0],neighbour.pos[0]],[particle.t,neighbour.t],linewidth=0.4,c='g')
        plt.pause(0.01)
    
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
        
    def _float_time(self, t: Union[int, float]) -> float:
        if isinstance(t, int):
            return t*self.structure.dt
        return t
        