import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

nx = 41
dx = 2./(nx-1)
nt = 10 
dt = .02 #seconds
c = 1 #assuming wavespeed of c = 1
x = np.linspace(0,2,nx) # space from 0 to 2 meteres in space

u = np.ones(nx) #numpy function ones()
lbound = np.where(x>=0.5)
ubound = np.where(x<=1) # returns indices where x <=1

bounds = np.intersect1d(lbound,ubound)
u[bounds] = 2 #setting u = 2 b/w x= 0.5 and x = 1

# CAN ALSO BE WRITTEN AS
#u[numpy.intersect1d(numpy.where(x >= 0.5), numpy.where(x <= 1))] = 2

'''
Q. Why is the square function sloping on the sides?
A. That's because the square wave function is injective and the mapping is one to one
'''
#Running the equation

for n in range(1,nt):
    un = u.copy()
    u[1:] = un[1:] - un[1:]*(dt/dx)*(un[1:]-un[0:-1])    
    u[0] = 1.0
    
plt.plot(x, u, color='#003366', ls='--', lw=3)
plt.ylim(0,2.5)
plt.show()
