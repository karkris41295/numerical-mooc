from math import sin, cos, ceil, log 
import numpy as np
from matplotlib import pyplot
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

# model parameters
g = 9.8 #ms^-2
v_t = 30.0 #trim velocity in ms^-1
C_D = 1./40 # coefficient of drag or D/L if C_L = 1  (THIS IN INCORRECT IN THE NOTEBOOK LESSON) it doesn't give float.
C_L = 1. # for convenience use C_L = 1

# C_D/C_L  = aerodynamic efficiancy
# set initial conditions 
v0 = v_t #initial trim velocity
theta0 = 0. #initial angle of trajectory
x0, y0 = 0., 1000. # coordinates

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

T = 100   # final time
dt_values = np.array([0.1, 0.05, 0.01, 0.005, 0.001])  # time increment values
u_values = np.empty_like(dt_values, dtype=np.ndarray) #creates empty array of arrays

for i, dt in enumerate(dt_values):
    N = int(T/dt) + 1   # no. of time steps
    t = np.linspace(0.0,T, N)  #t time descretization
    
    #initialize the array containing the solution for each time step
    u = np.empty((N,4)) # creates N x 4 array/matrix
    u[0] = np.array([v0,theta0,x0,y0]) #fill 1st element og list with init values
    
    # time simulation - Euler method
    for n in range(N-1): # N-1 because we already put in one value u[0]
        u[n+1] = euler_step(u[n], f, dt)
    
    #storing u values
    u_values[i] = u

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

def get_diffgrid(u_current, u_fine, dt):
    """Returns the difference between one grid and the fine one using L-1 norm.
    
    Parameters
    ----------
    u_current : array of float
        solution on the current grid.
    u_finest : array of float
        solution on the fine grid.
    dt : float
        time-increment on the current grid.
    
    Returns
    -------
    diffgrid : float
        difference computed in the L-1 norm.
    """
    
    N_current = len(u_current[:,0])
    N_fine = len(u_fine[:,0])
   
    if N_current != N_fine:
        grid_size_ratio = ceil(N_fine/N_current)+1 # ratio between grid sizes (no. of time steps)
        #ceil + 1?? WHYYYY
    else: grid_size_ratio = 1

    #difference between grids summed
    diffgrid = dt * np.sum( np.abs(\
            u_current[:,2]- u_fine[::grid_size_ratio,2])) # we're using the 3rd element in u (which is x0) wonder why not y0?
    
    return diffgrid

#compute difference between one grid solution and the finest one
diffgrid = np.empty_like(dt_values)

for i,dt in enumerate(dt_values):
    print('dt = {}'.format(dt))
    
    ### call the function get_diffgrid() ###
    diffgrid[i] = get_diffgrid(u_values[i],u_values[-1],dt)
    
pyplot.figure(figsize=(8,6))
pyplot.grid(1)
pyplot.xlabel('$\Delta t$' , fontsize=18)
pyplot.ylabel('$L_1$-norm of the grid differences', fontsize=18)
pyplot.axis('equal')
pyplot.loglog(dt_values[:-1], diffgrid[:-1], color='k',ls='-',lw=2,marker='o')
pyplot.show()

### ORDER OF CONVERGENCE ###
# p = log(f3-f2/f2-f1)/log(r) 
# f1 is finest mesh, f3 is the coarsest
# order of convergence measures how fast our solution converges to the real solution 
r = 2 #constant ratio
h = 0.001 #finest grid size

#Should probably write a function for this :P
dt_values2 = np.array([h, r*h, r**2*h])
u_values2 = np.empty_like(dt_values2, dtype =np.ndarray)

diffgrid2 = np.empty(2)
for i, dt in enumerate(dt_values2):
    N = int(T/dt) + 1   # no. of time steps
    t = np.linspace(0.0,T, N)  #t time descretization
    
    #initialize the array containing the solution for each time step
    u = np.empty((N,4)) # creates N x 4 array/matrix
    u[0] = np.array([v0,theta0,x0,y0]) #fill 1st element og list with init values
    
    # time simulation - Euler method
    for n in range(N-1): # N-1 because we already put in one value u[0]
        u[n+1] = euler_step(u[n], f, dt)
    
    #storing u values
    u_values2[i] = u

#calculate f2 - f1
diffgrid2[0] = get_diffgrid(u_values2[1], u_values2[0], dt_values2[1])

#calculate f3 - f2
diffgrid2[1] = get_diffgrid(u_values2[2], u_values2[1], dt_values2[2])

# calculate the order of convergence
p = (log(diffgrid2[1]) - log(diffgrid2[0])) / log(r)

print('The order of convergence is p = {:.3f}'.format(p));
#p = 1.014, that means numerator of p is almost equal to mesh refinement ratio (log2)