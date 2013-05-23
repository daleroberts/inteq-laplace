## Solve the Dirichlet problem for the Laplace equation on 
## a planar domain where the boundary is a smooth simple 
## closed curve with a C^2 parametrisation using the 
## boundary integral equation method.
##
## Dale Roberts <dale.o.roberts@gmail.com>

import numpy as np
import scipy.linalg as la

import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.tri as tri

## Set the parameters

n_boundary = 128 
n_quadrature = n_boundary
n_angles = n_boundary
n_radii = n_boundary
min_radius = 0.01
max_radius = 0.99
plot_contours = False
n_levels = 128
colors = cm.prism

## Now define the boundary surface parametrisation
##
## \[    r(t) = (\xi(t), \eta(t)),  0 \le t \le L     \]
##
## with $r \in C^2[0,L]$ and $|r'(t)| \ne 0$ for $0 \le t \le L$.

L = 2.0 * np.pi # angles from 0 to L

# ellipse boundary

a = 1 # half major axis
b = 1 # half minor axis

def r(t):
    # ellipse boundary
    x = a * np.cos(t)
    y = b * np.sin(t)
    return x, y

## We solve the integral equation:
##
## \[ - \pi \rho(t) + \int_0^L k(t,s) \rho(s) \, ds = f(t)  \]
##
## for $0 \le t \le L$, where the kernel k(t,s) is given by
##
## \[ 
## k(t,s) = 
##      \frac{
##       \eta'(s)[\xi(t) - \xi(s)] - \xi'(s)[\eta(t) - \eta(s)],
##       [\xi(t)- \xi(s)]^2 + [\eta(t) - \eta(s)]^2
##      }
## \]
##
## for $s \ne t$, and
##
## \[ 
## k(t,t) = 
##      \frac{
##       \eta'(s)\xi''(t) - \xi'(s)\eta''(t),
##       2 [\xi'(t)^2 + \eta'(s)^2]
##      }
## \]


def k(r,s):
    # ellipse boundary
    theta = (r+s)/2.0
    cost = np.cos(theta)
    sint = np.sin(theta)
    sig  = np.sqrt(a**2.0 * sint**2.0 + b**2.0 * cost**2.0)
    return -a * b / (2.0 * sig**2.0)


## and the boundary data $f(t)$ is given by
## $f(t) \equiv f(r(t))$.

def f(r):
    return np.sin(10*r)


## Assemble the linear system
##
## \[
##
## - \pi \rho_n(t_i) + h \sum_{j=1}^n k(t_i,t_j)\rho_n(t_j) = f(t_i)  
##
## \]
##
## where n = n_boundary, h = L/n, t_j = j h.


# Sample the angles for the boundary discretisation

t, h = np.linspace(0, L, n_boundary, endpoint=False, retstep=True)

# Assemble matrix (dense!)

A = np.zeros((n_boundary,n_boundary))

for i in range(n_boundary):
    for j in range(n_boundary):
        A[i,j] = k(t[i],t[j])

A = -np.pi * np.eye(n_boundary) + h * A

# Assemble right-hand side

f_n = f(t)

## Solve for the approximation of the kernel $\rho_n$.

rho_n = la.solve(A,f_n)

## We can approximate the kernel $\rho(t)$ at any t using 
## interpolation.

def rho_int(s):
    # Nystrom interpolation to obtain value at arbitrary s
    K = h*np.dot(k(s,t),rho_n)
    return 1.0/np.pi * (-f(s) + K)

if n_quadrature != n_boundary:
    rho = np.array([rho(tau) for tau in T]).flatten()
else:
    rho = rho_n

## We now need to evaluate the double layer potential
##
## u(x,y) = \int_0^L M(x,y,s) \rho(s)\,ds
##
## where
##
## \[
##   M(x,y,s) = 
##   \frac{
##    -\eta'(s)[\xi(s) - x] + \xi'(s) [\eta(s) - y],
##    [\xi(s) - x]^2 + [\eta(s) - y]^2
##   }
## \]
##
## We use a trapezoidal rule.

def M(x,y,s):
    coss = np.cos(s)
    sins = np.sin(s)
    numer = - coss*(coss - x) - sins*(sins - y)
    denom = (coss - x)**2.0 + (sins-y)**2.0
    return numer/denom

## Sample the angles for the quadrature

T, H = np.linspace(0, L, n_quadrature, endpoint=False, retstep=True)

def u(x,y):
    # solution given by trapesoidal rule
    return H * np.dot(M(x,y,T),rho)

## We now plot the solution

# First sample the x and y coordinates of the points we want
# to evaluate the solution u(x,y) at

radii = np.linspace(min_radius, max_radius, n_radii)

angles = np.linspace(0, L, n_angles, endpoint=False)
angles = np.repeat(angles[...,np.newaxis], n_radii, axis=1)
angles[:,1::2] += np.pi/n_angles

X = a*(radii*np.cos(angles)).flatten()
Y = b*(radii*np.sin(angles)).flatten()

Z = []
for x,y in zip(X,Y):
    Z.append(u(x,y))
Z = np.array(Z)

# unit levelsets for contour plot

levelsets = np.linspace(np.floor(np.min(Z))-1,np.ceil(np.max(Z))+1, n_levels)

# clear plot

plt.clf()

# create mesh

mesh = tri.Triangulation(X, Y)

# Mask off unwanted triangles.

xmid = X[mesh.triangles].mean(axis=1)
ymid = Y[mesh.triangles].mean(axis=1)
mask = np.where(xmid*xmid + ymid*ymid < min_radius*min_radius, 1, 0)
#mesh.set_mask(mask)

plt.gca().set_aspect('equal')

if plot_contours:
    plt.tricontourf(mesh, Z, levels=levelsets, cmap=colors)
    plt.colorbar()
    plt.tricontour(mesh, Z, levels=levelsets, cmap=colors)
else:
    plt.tripcolor(mesh, Z)

plt.show()
