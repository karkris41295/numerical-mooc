# PAPER PLANE MOD OF NBK 3
from math import sin, cos
import numpy as np
from matplotlib import pyplot
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

# model parameters
g = 9.8 #ms^-2
v_t = 4.9 #trim velocity in ms^-1
C_D = 1./5 # coefficient of drag or D/L if C_L = 1  (THIS IN INCORRECT IN THE NOTEBOOK LESSON) it doesn't give float.
C_L = 1. # for convenience use C_L = 1

# C_D/C_L  = aerodynamic efficiancy
# set initial conditions 
v0 = 1000 #initial velocity
theta0 = 0. #initial angle of trajectory
x0, y0 = 0., 3. # coordinates

def f(u):
    """Returns the right-hand side of the phugoid system of equations.
    
    Parameters
    ----------
    u : array of float
        array containing the solution at time n.
        
    Returns
    -------
    dudt : array of float
        array containing the RHS given u.
    """
    
    v, theta, x, y = u   # passing list to a tuple
    return np.array([-g*sin(theta) -(C_D/C_L) * (g/v_t**2)*(v**2), -g*cos(theta)/v + (g/v_t**2)*v, v*cos(theta), v*sin(theta)])
    
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

T = 10   # final time
dt = 0.001  # time increment values
N = int(T/dt) +1
u = np.empty((N,4))

t = np.linspace(0.0,T, N)  #t time descretization
    
#initialize the array containing the solution for each time step
u = np.empty((N,4)) # creates N x 4 array/matrix
u[0] = np.array([v0,theta0,x0,y0]) #fill 1st element of list with init values
  
# time simulation - Euler method
for n in range(N-1): # N-1 because we already put in one value u[0]
    u[n+1] = euler_step(u[n], f, dt)
    if np.abs(u[n+1,3] - 0.00) < 0.000003:
        break    
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
    # plotting the trajectory, let's get the x and y values out!
   x, y = u[:,2], u[:,3]
    
    #visualization of the path
   pyplot.figure(figsize = (8,6))
   pyplot.grid(True)
   pyplot.xlabel(r'x', fontsize = 18)
   pyplot.ylabel(r'y', fontsize = 18)
   pyplot.title('Glider trajectory, flight time = %.2f' % T, fontsize=18)
   pyplot.plot(x,y,'k-', lw=2)
   
   pyplot.show()

show_ind_plot(u)
