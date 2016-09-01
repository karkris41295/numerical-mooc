import numpy                       
from matplotlib import pyplot    
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

nx = 100
dx = 2./(nx-1)
nt = 20    
sigma = 0.5 
c = 1
    
dt = sigma*dx  # CFL MOD
x = numpy.linspace(0,2,nx)

u = numpy.ones(nx)
lbound = numpy.where(x >= 0.5)
ubound = numpy.where(x <= 1)
u[numpy.intersect1d(lbound, ubound)]=2  

un = numpy.ones(nx)

fig = pyplot.figure(figsize=(11,8))
ax = pyplot.axes(xlim=(0,2), ylim=(0,3))
line = ax.plot([],[],color = '#003366', ls ='--',lw=3)[0]
pyplot.xlabel('x', fontsize = 18)
pyplot.ylabel('u', fontsize = 18)
pyplot.title('Wave Equation: Visualized!', fontsize=18) 

def wave(i):
    
    line.set_data(x,u) 
    
    un = u.copy() 
    u[1:] = un[1:] -c*dt/dx*(un[1:] -un[0:-1]) 
        
from matplotlib import animation

ani = animation.FuncAnimation(fig, wave,interval=100)
pyplot.show()