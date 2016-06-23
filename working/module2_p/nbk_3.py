import numpy                       
from matplotlib import pyplot    
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

nx = 41
dx = 2./(nx-1)
nt = 20
nu = 0.3 #value of viscosity
sigma = .2
dt = sigma * dx**2/nu

x = numpy.linspace(0,2,nx)
ubound = numpy.where(x >= 0.5)
lbound = numpy.where(x <= 1)
nt = 50

u = numpy.ones(nx)      
u[numpy.intersect1d(lbound,ubound)] = 2  

un = numpy.ones(nx) 

fig = pyplot.figure(figsize=(11,8))
ax = pyplot.axes(xlim=(0,2), ylim=(1,3))
line = ax.plot([],[],color = '#003366', ls ='--',lw=3)[0]
pyplot.xlabel('x', fontsize = 18)
pyplot.ylabel('u', fontsize = 18)
pyplot.title('Diffusion Equation: Visualized!', fontsize=18)

def diffusion(i):
    line.set_data(x,u)

    un = u.copy() 
    u[1:-1] = un[1:-1] + nu*dt/dx**2*\
            (un[2:] - 2*un[1:-1] + un[0:-2])
    
from matplotlib import animation

ani = animation.FuncAnimation(fig, diffusion,interval=100)
pyplot.show()