#re-attempting green-light problem with the Finite Volume Method

from nbk_1 import rho_green_light
from nbk_4 import godunov
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

#Basic initial condition parameters
#defining grid size, time steps
nx = 81
nt = 40
dx = 4.0/(nx-1)

x = np.linspace(0,4,nx)

rho_max = 10.
u_max = 1.
rho_light = 10. #at 10 causes both left and right waves, for stablity, set to 5

rho = rho_green_light(nx, rho_light)
sigma = 0.8  #CFL number, gets wobbly at 1.2 on
dt = sigma*dx

rho_n = godunov(rho, nt, dt, dx, rho_max, u_max)

from matplotlib import animation

fig = plt.figure();
ax = plt.axes(xlim=(0,4),ylim=(-.5,11.5),xlabel=('Distance'),ylabel=('Traffic density'));
line, = ax.plot([],[],color='#003366', lw=2);

def animate(data):
    x = np.linspace(0,4,nx)
    y = data
    line.set_data(x,y)
    return line,

anim = animation.FuncAnimation(fig, animate, frames=rho_n, interval=50)
plt.show()
