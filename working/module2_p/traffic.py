import numpy
import sympy
from matplotlib import pyplot
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

v_max = 80. #km/hr
L = 11. #km
rho_max = 250 #cars/km
nx = 51
nt = 30
dx = L/(nx-1)
dt = .001 #hrs

from sympy.utilities.lambdify import lambdify # converts expression into a funtio

rho, flux = sympy.symbols('rho flux')
flux = v_max*rho*(1-rho/rho_max)
f_lamb = lambdify(rho, flux)

x = numpy.linspace(0,L,nx)
rho0 = numpy.ones(nx)*10
rho0[10:20] = 50
rho0[0] = 20.

fig = pyplot.figure(figsize=(11,8))
ax = pyplot.axes(xlim=(0,11), ylim=(0,5000))
line = ax.plot([],[],color = '#003366', ls ='--',lw=3)[0]

def traffic(i): 
    
    rn = rho0.copy()
    rho0[0] = 20. 
    f = numpy.asarray([f_lamb(r0) for r0 in rho0])
    line.set_data(x,f)
    rho0[1:] = rn[1:] -dt/dx*(f[1:] -f[0:-1])

from matplotlib import animation

ani = animation.FuncAnimation(fig, traffic,interval=100)

pyplot.show()  
