#Simulation of an Inviscid Burgers' Equation in 1D

import numpy as np
from matplotlib import pyplot
from matplotlib import animation
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

def u_initial(nx):
    u = np.ones(nx)
    u[int(nx/2):] = 0.
    return u
    
def computeF(u):
    return u**2/2.
    
def maccormack(u, nt, dt, dx):
    un = np.zeros((nt,len(u)))
    un[:] = u.copy()
    ustar = u.copy()    

    for n in range(1,nt):
        F = computeF(u)
        #Trying out some numerical damping!
        ustar[1:-1] = u[1:-1] - (dt/dx)*(F[2:] - F[1:-1]) + e*(u[2:] - 2*u[1:-1] + u[:-2])
        #...The result is crazy
        Fstar = computeF(ustar)
        un[n,1:-1] = .5*(u[1:-1] + ustar[1:-1] - (dt/dx)*(Fstar[1:-1]-Fstar[:-2]))
        u = un[n].copy()

    return un

#######RUN################
nx = 81
nt = 70
dx = 4.0/(nx-1)

def animate(data):
    x = np.linspace(0,4,nx)
    y = data
    line.set_data(x,y)
    return line,

u = u_initial(nx)
sigma = .5
dt = sigma*dx
e = 0.3 # numerical damping factor

un = maccormack(u,nt,dt,dx)

fig = pyplot.figure();
ax = pyplot.axes(xlim=(0,4),ylim=(-.5,2),xlabel=('x'),ylabel=('u'));
line, = ax.plot([],[],lw=2);

anim = animation.FuncAnimation(fig, animate, frames=un, interval=42)