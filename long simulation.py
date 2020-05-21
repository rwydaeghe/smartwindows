from smartwindow import *
import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.close('all')
k=[]
print('initializing')
structure = Structure((20e-5, 5e-5))
simulation=Simulation(structure)

print('adding particles')
structure.add_gaussian_particle_cloud(N=75, avg_pos=(10e-5,2.5e-5), var_pos=(1e-5,1e-5),seed=0)
values=[]
n=20
for i in range(n):
    chance=0.5+(0.5)*((n-i)/n)**(n/5)
    chance=73/100
    a=-40
    values.append([[a*(1-chance),a*chance,0,0],[0,0,a*(1-chance),a*chance],[a*chance,a*(1-chance),0,0],[0,0,a*chance,a*(1-chance)]])
periods = []
structure.load_fields()
cycle_time=1.0 #200 iteraties
simulation.dt['electrode_cycle']=1e0
simulation.dx_max=5e-2 #infinite

print('starting simulation')
print(cycle_time*n)
simulation.run(cycle_time*n,
               animate=False,
               animate_events=False,
               with_field='nothing',
               with_arrows=False,
               electrode_values=values)
for particle in structure.particles:
    periods.append(particle.structure_period)
result= np.bincount(np.array(periods)+n)
result[n-1]+=1500
plt.close('all')
k.append(list(result))
