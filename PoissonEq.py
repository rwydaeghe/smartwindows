#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 19:36:58 2020

@author: hanne
"""


from dolfin import *
import time

t = time.clock()

# Classes for different electrodes
class E1(SubDomain):
  def inside(self, x, on_boundary):
    return (near(x[1], 0.0) and between(x[0], (0.0,4.0e-5)))

class E2(SubDomain):
  def inside(self, x, on_boundary):
    return (near(x[1], 0.0) and between(x[0], (8.0e-5,12.0e-5))) 

class E3(SubDomain):
  def inside(self, x, on_boundary):
    return (near(x[1], 5.0e-5) and between(x[0], (4.0e-5,8.0e-5)))

class E4(SubDomain):
  def inside(self, x, on_boundary):
    return (near(x[1], 5.0e-5) and between(x[0], (12.0e-5,16.0e-5)))

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
        y[0] = x[0] - 16.0e-5
        y[1] = x[1]

# Create periodic boundary condition
pbc = PeriodicBoundary()

# Create mesh and finite element
p0 = Point(0.0,0)
p1 = Point(16.0e-5,5.0e-5)
mesh = RectangleMesh(p0,p1,1600,500)
V = FunctionSpace(mesh, "CG", 1, constrained_domain=pbc)
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0.0)

# Define Dirichlet boundary conditions for the four electrodes
bcs = [DirichletBC(V, 1.0e-5, e1), DirichletBC(V, 1.0e-5, e2), DirichletBC(V, 1.0e-5, e3), DirichletBC(V, 1.0e-5, e4),]

a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bcs)
print(time.clock()-t)

# Plot solution
c = plot(u)
plt.colorbar(c)