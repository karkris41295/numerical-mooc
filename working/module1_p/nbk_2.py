import matplotlib.pyplot as plt
import numpy as np
################################################################################
"""
INITIAL VALUES
"""
T = 100.0 #initial time
dt_values = np.array([0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]) #time steps 
z_values = np.empty_like(dt_values, dtype=np.ndarray) #creates an empty array with shape dt_values but type array 
#above fn creates array of arrays or at least makes a placeholder so we can copy one later

z0 = 100. #altitude
b0 = 0#10. # upward velocity resulting from gust
zt = 100.
g = 9.81

################################################################################
"""
FUNCTIONS
"""
 
def plot_num_sol(z, t):     #method to plot numerical solution z vs t
    """Sets plot with Numerical and Analytical solutions.
    
    Parameters
    ----------
    z : array of float
        numerical solution.
    t : array of float
        points on time grid.

    """
    plt.figure(figsize =(10,5)) #set plot size
    plt.ylim(40,160) #y-axis plot limits
    plt.tick_params(axis = 'both', labelsize = 14) #fontsize on axes
    plt.xlabel('t', fontsize=14) #x label
    plt.ylabel('z', fontsize=14) #y label
    plt.plot(t,z)
    z_exact = b0*(zt/g)**.5*np.sin((g/zt)**.5*t) +\
          (z0-zt)*np.cos((g/zt)**.5*t) + zt
    plt.plot(t,z_exact)
    plt.legend(['Numerical Solution', 'Analytical Solution'])
    
def get_error(z, dt):
    """Returns the error relative to analytical solution using L-1 norm.
    
    Parameters
    ----------
    z : array of float
        numerical solution.
    dt : float
        time increment.
        
    Returns
    -------
    err : float
        L_{1} norm of the error with respect to the exact solution.
    """
    N = len(z)
    t = np.linspace(0.0,T,N)
    
    z_exact = b0*(zt/g)**.5*np.sin((g/zt)**.5*t) + (z0-zt)*np.cos((g/zt)**.5*t) + zt             
          
    error = np.sum(np.abs(z-z_exact)) #find error (abs of difference between z and z exact vector and sum them up)
    return dt *error # calculation of 'norm' so it's weighted by dt (I think, idk)
    #it's not weight, it's the sum of all values * dt to give the norm, see the definition of norm :P - future kartik 
    
def euler(u, dt, N):     #Euler's Method 
    """Returns an array of z(solution) for given time grid using Euler's Method
    
    Parameters
    ----------
    u : array of float
        state vector of flight [z,z'].
    dt : float
        time increment.
    N  : float
        length of time 'grid'
        
    Returns
    -------
    z : array of float
        Array of z positions with respect to time.
    """
    #initialize array with z (elevation) values (height down from 0 potential)
    z = np.zeros(N)
    for n in range(1, N):
        u = u + dt*np.array([u[1], g*(1-u[0]/zt)])
        z[n] = u[0]  #0th element of vector is position(s)
    return z
        
    
################################################################################
"""
MAIN
"""

for i,dt in enumerate(dt_values): #creates a list of tuples of (i, dt). i is used later in this block
    t = np.arange(0.0, 100, dt) #time grid
    N = len(t)
    u = np.array([z0,b0])
    z_values[i] = euler(u,dt,N) #copies the values found by euler's method into the z_values array

error_values = np.empty_like(dt_values)

for i, dt in enumerate(dt_values):  #adds up all errors
    ### call the function get_error() ###
    error_values[i] = get_error(z_values[i], dt) 

plt.figure(figsize=(10, 6))
plt.tick_params(axis='both', labelsize=14) #increase tick font size
plt.grid(True)                         #turn on grid lines
plt.xlabel('$\Delta t$', fontsize=16)  #x label
plt.ylabel('Error', fontsize=16)       #y label
plt.loglog(dt_values, error_values, 'ko-')  #log-log plot
plt.axis('equal')                      #make axes scale equally;
        
plt.show()