#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 17:10:46 2020

@author: hanne
"""


from dolfin import *
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, CubicTriInterpolator, LinearTriInterpolator
import numpy as np
import pickle

eps_vac = 8.854e-12
eps_0 = 2
eps_1 = 5
tol = 1E-14



y_b = 0.0e-5
y_t = 5.0e-5
t = 1.5e-5   
#t = 0                                                                     # dikte glas
p0 = Point(0.0,y_b - t)
p1 = Point(20.0e-5,y_t + t)
#domain = Rectangle(p0,p1)
#sub = Rectangle(Point(0.0,0.0),Point(20.0e-5,y_t))
#domain.set_subdomain(2,sub)
#mesh = generate_mesh(domain,200)
y_cells = int(50 + 2*t*10**6)
mesh = RectangleMesh(p0,p1,200,y_cells)

class Omega_0(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] <= y_t + tol and x[1] >= y_b - tol
    
class Omega_1_1 (SubDomain):
    def inside(self, x, on_boundary):
        return x[1] >= y_b - t - tol and x[1] <= y_b + tol
class Omega_1_2 (SubDomain):
    def inside(self, x, on_boundary):
        return x[1] >= y_t - tol and x[1] <= y_t + t + tol  

materials = MeshFunction('size_t', mesh, mesh.topology().dim())

subdomain_0 = Omega_0()
subdomain_1_1 = Omega_1_1()
subdomain_1_2 = Omega_1_2()
subdomain_0.mark(materials, 0)
subdomain_1_1.mark(materials, 1)
subdomain_1_2.mark(materials, 1)

#plt.figure()
#plot(materials)
#plt.title('Different Subdomains')
#plt.xlabel('x [m]')
#plt.ylabel('y [m]')
#plt.show()


#tol = 5.75e-7

class E1(SubDomain):
  def inside(self, x, on_boundary):
    return near(x[1], 0.0) and between(x[0], (0.0,5.0e-5)) 

class E2(SubDomain):
  def inside(self, x, on_boundary):
    return near(x[1], 0.0) and between(x[0], (10.0e-5,15.0e-5)) 

class E3(SubDomain):
  def inside(self, x, on_boundary):
    return near(x[1], 5.0e-5) and between(x[0], (5.0e-5,10.0e-5)) 

class E4(SubDomain):
  def inside(self, x, on_boundary):
    return near(x[1], 5.0e-5) and between(x[0], (15.0e-5,20.0e-5)) 

# Initialize sub-domain instances
e1 = E1()
e2 = E2()
e3 = E3()
e4 = E4()

electrodes = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
electrodes.set_all(0)
e1.mark(electrodes,1)
e2.mark(electrodes,2)
e3.mark(electrodes,3)
e4.mark(electrodes,4)

dx = Measure('dx', domain=mesh, subdomain_data = materials)
ds = Measure('ds', domain=mesh, subdomain_data = electrodes)
        
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
        
V = FunctionSpace(mesh,'CG',1,constrained_domain=pbc)
def DirBoundary(v1=0, v2=0, v3=0, v4=0):
    bcs = [DirichletBC(V, v1, electrodes,1), DirichletBC(V, v2, electrodes,2), DirichletBC(V, v3, electrodes,3), DirichletBC(V, v4, electrodes,4)]
    return bcs







u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0.0)
g = Constant(0.0)


a = eps_vac*eps_0*dot(grad(u), grad(v))*dx(0) + eps_vac*eps_1*dot(grad(u), grad(v))*dx(1)
#a = dot(grad(u), grad(v))*dx
L = f*v*dx

A, b = assemble_system(a, L, DirBoundary())

u = Function(V)

point_sources = np.zeros(201*(y_cells+1)*201*51).reshape(201,51,201*(y_cells+1))

for x in range(201):
    for y in range(51):
        p = Point(x*10**(-6),y*10**(-6))
        delta = PointSource(V,p,1)
        delta.apply(b)
        solve(A, u.vector(), b)
        z = u.compute_vertex_values(mesh)
        point_sources[x,y] = z
        delta_remove = PointSource(V,p,-1)
        delta_remove.apply(b)






u1 = Function(V)
u2 = Function(V)
u3 = Function(V)
u4 = Function(V)
solve(a == L, u1, DirBoundary(v1=1))
solve(a == L, u2, DirBoundary(v2=1))
solve(a == L, u3, DirBoundary(v3=1))
solve(a == L, u4, DirBoundary(v4=1))

#plt.figure()
#c = plot(u1)
#plt.colorbar(c)
#plt.title('Electric Potential')
#plt.xlabel('x [m]')
#plt.ylabel('y [m]')
#plt.show()


#plot(mesh)

# Saving variables as pickle file

mesh = u1.function_space().mesh()
z1 = u1.compute_vertex_values(mesh)
z2 = u2.compute_vertex_values(mesh)
z3 = u3.compute_vertex_values(mesh)
z4 = u4.compute_vertex_values(mesh)
x = mesh.coordinates()[:,0]
y = mesh.coordinates()[:,1]
cells = mesh.cells()
#
## making triangulation
triang = Triangulation(x,y,cells)
##
with open ('Variables/voltages_with_glass.pkl','wb') as f:
    pickle.dump([z1, z2, z3, z4],f)
    
with open ('Variables/triangulation_with_glass.pkl','wb') as f:
    pickle.dump(triang,f)

#trifinder = triang.get_trifinder()
##
#tci = LinearTriInterpolator(triang,z1)
#(Ex,Ey) = tci.gradient(triang.x,triang.y)
#E = -np.array([Ex,Ey])
#
#x = np.linspace(0,20e-5,200)
#y = np.linspace(y_b - t,y_t + t,y_cells)
#X,Y = np.meshgrid(x,y)
#
#def get_field_here(field, pos: np.ndarray) -> np.ndarray:
#        tr = trifinder(*pos)                               
#        i = triang.triangles[tr]   
#        v0 = np.array([triang.x[i[0]],triang.y[i[0]]])  
#        v1 = np.array([triang.x[i[1]],triang.y[i[1]]])
#        v2 = np.array([triang.x[i[2]],triang.y[i[2]]])
#        norm = np.array([np.linalg.norm(v0-pos),np.linalg.norm(v1-pos),np.linalg.norm(v2-pos)])
#        j = np.argmin(norm)                                                     
#        v = i[j]
#        fieldx = np.array(field[0])
#        fieldy = np.array(field[1])
#        return np.array([fieldx[v], fieldy[v]])
#
#def get_E(pos: np.ndarray=None):
#        if pos.any()==None:
#            return E
#        else:
#            return get_field_here(E,pos)
#
#Ex=np.array([get_E(np.array([x,y]))[0] for (x,y) in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
#Ey=np.array([get_E(np.array([x,y]))[1] for (x,y) in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
#
#plt.figure()
#plt.streamplot(X,Y,Ex,Ey,density=3)
#plt.title('Streamplot of Electric Field')
#plt.xlabel('x [m]')
#plt.ylabel('y [m]')
#plt.axis([0, 20e-5, y_b - t, y_t + t])
#plt.show()
#
#plt.figure()
#plt.quiver(X,Y,Ex,Ey)
#plt.title('Vectorplot of Electric Field')
#plt.xlabel('x [m]')
#plt.ylabel('y [m]')
#plt.show()


