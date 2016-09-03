# -*- coding: utf-8 -*-
# ROCKET MOD OF NBK 3
from math import pi
import numpy as np
from matplotlib import pyplot
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

# model parameters
g = 9.81 #ms^-2
ms = 50 #kg
p = 1.091 #kgm^-3
A = pi * 0.5**2
ve = 325 #exhaust speed in ms^-1
C_D = 0.15 # coefficient of drag or D/L if C_L = 1 
mp = 100 #kg iniitial propellant mass
mp_dot = 20 #initial rocket propulsion rate

# C_D/C_L  = aerodynamic efficiancy
# set initial conditions 
h0 = 0
v0 = 0
#initial height and velocity
def f(u):
    """Returns the right-hand side of the rocket equations.
    
    Parameters
    ----------
    u : array of float
        array containing the solution at time n.
        
    Returns
    -------
    dudt : array of float
        array containing the RHS given u.
    """
    
    h, v = u   # passing array to a tuple (I have no clue what past Kartik was talking about)
    return np.array([v, -g + (mp_dot*ve)/(ms + mp) - (.5 * p * v * np.abs(v)*A*C_D)/(ms+mp)])

def euler_step(u, f, dt):
    """Returns the solution at the next time-step using Euler's method.
    
    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equation.
    dt : float
        time-increment.
    
    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """
    
    return u + dt * f(u)

T = 50 # final time
dt = 0.1  # time increment values
t = np.arange(0,T+dt,dt)  #t time descretization
N = len(t)

#initialize the array containing the solution for each time step
u = np.empty((N,2)) # creates N x 2 array/matrix
u[0] = np.array([h0, v0]) #fill 1st element of list with init values
  
# time simulation - Euler method
for n in range(N-1): #because we're calculating n+1
    # mp update condition
    if t[n+1] >= 5.000:
        mp_dot = 0
    mp = mp - mp_dot * dt 
    u[n+1] = euler_step(u[n], f, dt)

    
def show_ind_plot(u):
   """shows individual plot of a solution given u.
    
    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    
    Returns
    -------
    A plot of the numerical solution.
    """ 
   # plotting the trajectory, let's get h        
   h = u[:,0]
   
   #Finding where the height becomes negative
   idx_negative = np.where(h<0.0)[0]
   if len(idx_negative)==0:
       idx_ground = N-1
       print ('Euler integration has not touched ground yet!')
   else:
       idx_ground = idx_negative[0]   
   
   #visualization of the path
   pyplot.figure(figsize = (8,6))
   pyplot.grid(True)
   pyplot.xlabel(r't', fontsize = 18)
   pyplot.ylabel(r'h', fontsize = 18)
   pyplot.title('Flight time = %.2f' % T, fontsize=18)
   pyplot.plot(t[:idx_ground],h[:idx_ground],'k-', lw=2)
   print 'max height =' + str(np.amax(h))
   print u[idx_ground,1]
   pyplot.show()

show_ind_plot(u)