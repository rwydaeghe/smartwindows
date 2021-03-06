#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 19:36:58 2020

@author: hanne
"""


from dolfin import *
import time
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, CubicTriInterpolator
import numpy as np
import pickle
import random as r

t1 = time.clock()

# Classes for different electrodes
class E1(SubDomain):
  def inside(self, x, on_boundary):
    return (near(x[1], 0.0) and between(x[0], (0.0,5.0e-5)))

class E2(SubDomain):
  def inside(self, x, on_boundary):
    return (near(x[1], 0.0) and between(x[0], (10.0e-5,15.0e-5))) 

class E3(SubDomain):
  def inside(self, x, on_boundary):
    return (near(x[1], 5.0e-5) and between(x[0], (5.0e-5,10.0e-5)))

class E4(SubDomain):
  def inside(self, x, on_boundary):
    return (near(x[1], 5.0e-5) and between(x[0], (15.0e-5,20.0e-5)))

# Initialize sub-domain instances
e1 = E1()
e2 = E2()
e3 = E3()
e4 = E4()

# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

    # Map right boundary to left boundary 
    def map(self, x, y):
        y[0] = x[0] - 20.0e-5
        y[1] = x[1]

# Create periodic boundary condition
pbc = PeriodicBoundary()



# Create mesh and finite element
p0 = Point(0.0,0.0)
p1 = Point(20.0e-5,5.0e-5)
mesh = RectangleMesh(p0,p1,200,50)
V = FunctionSpace(mesh, "CG", 1, constrained_domain=pbc)
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0.0)
g = Constant(0.0)

# Define Dirichlet boundary conditions for the four electrodes
def DirBoundary(v1=0, v2=0, v3=0, v4=0):
    bcs = [DirichletBC(V, v1, e1), DirichletBC(V, v2, e2), DirichletBC(V, v3, e3), DirichletBC(V, v4, e4),]
    return bcs


a = dot(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

A, b = assemble_system(a, L, DirBoundary())

# Compute solution
#u1 = Function(V)
#u2 = Function(V)
#u3 = Function(V)
#u4 = Function(V)
u = Function(V)
solve(a == L, u, DirBoundary(v1=1))
#solve(a == L, u1, DirBoundary(v1=1))
#solve(a == L, u2, DirBoundary(v2=1))
#solve(a == L, u3, DirBoundary(v3=1))
#solve(a == L, u4, DirBoundary(v4=1))

#point_sources = np.zeros(105083001).reshape(201,51,10251)
#
#for x in range(201):
#    for y in range(51):
#        p = Point(x*10**(-6),y*10**(-6))
#        delta = PointSource(V,p,1)
#        delta.apply(b)
#        solve(A, u.vector(), b)
#        z = u.compute_vertex_values(mesh)
#        point_sources[x,y] = z
#        delta_remove = PointSource(V,p,-1)
#        delta_remove.apply(b)
    


print(time.clock()-t1)

# Plot solution
c = plot(u)
plt.colorbar(c)


# storing solution as arrays
#mesh = u1.function_space().mesh()
#z1 = u1.compute_vertex_values(mesh)
#z2 = u2.compute_vertex_values(mesh)
#z3 = u3.compute_vertex_values(mesh)
#z4 = u4.compute_vertex_values(mesh)
#x = mesh.coordinates()[:,0]
#y = mesh.coordinates()[:,1]
#t = mesh.cells()
#
## making triangulation
#triang = Triangulation(x,y,t)



# saving variables as pickle file
#with open ('Variables/voltages_small.pkl','wb') as f:
#    pickle.dump([z1,z2,z3,z4],f)
#    
#with open ('Variables/triangulation_small.pkl','wb') as f:
#    pickle.dump(triang,f)
    
