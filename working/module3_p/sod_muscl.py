#THIS CODE DOESN'T WORK YET :(

import numpy 
from matplotlib import pyplot
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

def initial(nx):
    """Computes "sod shock tube" initial condition with shock

    Parameters
    ----------
    x         : array of floats
        Points on grid

    Returns
    -------
    u: array of arrays of size 3
        Array with initial values of u vector
        Vector values are [rho, u, p] in kg/m**3, m/s, kN/m**2
    """
    i = numpy.empty((nx,3))
    mid = 40
    i[:mid,] = numpy.array([1., 0., 1e5]) 
    i[mid:,] = numpy.array([0.125, 0., 1e4])
    return i

def computeU(i):
    """Computes u vector

    Parameters
    ----------
    u    : array of arrays
        Array with u vector at every point x
        
    Returns
    -------
    F : array of arrrays
        Array with flux at every point x
    """
    i1 = i[:,0]
    i2 = i[:,1]
    i3 = i[:,2]
    u = numpy.array([i1, i1 * i2, (i3/(gamma-1.)+.5*i1*i2**2)]).T
    return u
    
def computeF(u):
    """Computes flux 

    Parameters
    ----------
    u    : array of arrays
        Array with u vector at every point x
        
    Returns
    -------
    F : array of arrrays
        Array with flux at every point x
    """
    u1 = u[:,0]
    u2 = u[:,1]
    u3 = u[:,2]
    f = numpy.array([u2,\
                  u2**2/u1+(gamma-1)*(u3-u2**2/(2*u1)),\
                  u2/u1*(u3+(gamma-1)*(u3-u2**2/(2*u1)))]).T
    return f
    
def final(u):
    """Breaks down the u vector to give back a final vector 

    Parameters
    ----------
    u    : array of arrays
        Array with u vector at every point x
        
    Returns
    -------
    final : array of arrrays
        Array with flux at every point x
        Vector values are [rho, u, p] in kg/m**3, m/s, kN/m**2
    """
    u1 = u[:,0]
    u2 = u[:,1]
    u3 = u[:,2]
    f = numpy.array([u1, u2/u1, (gamma-1)*(u3-u1*(u2/u1)**2)]).T
    return f

def minmod(e, dx):
    """
    Compute the minmod approximation to the slope
    
    Parameters
    ----------
    e : array of float 
        input data
    dx : float 
        spacestep
    
    Returns
    -------
    sigma : array of float 
            minmod slope
    """
    
    sigma = numpy.zeros_like(e)
    de_minus = numpy.ones_like(e)
    de_plus = numpy.ones_like(e)
    
    de_minus[1:] = (e[1:] - e[:-1])/dx
    de_plus[:-1] = (e[1:] - e[:-1])/dx
    
    # The following is inefficient but easy to read
    for i in range(1, len(e)-1):
        if (de_minus[i] * de_plus[i] < 0.0):
            sigma[i] = 0.0
        elif (numpy.abs(de_minus[i]) < numpy.abs(de_plus[i])):
            sigma[i] = de_minus[i]
        else:
            sigma[i] = de_plus[i]
            
    return sigma
    
def muscl(u, nt, dt, dx):
    """ Computes the solution with the MUSCL scheme using the Lax-Friedrichs flux,
    RK2 in time and minmod slope limiting.
    
    Parameters
    ----------
    u      : array of floats
            Density at current time-step
    nt     : int
            Number of time steps
    dt     : float
            Time-step size
    dx     : float
            Mesh spacing

    Returns
    -------
    u      : array of floats
            Density after nt time steps at every point x
    """
    
    #initialize our results array with dimensions nt by nx
    u_n = numpy.zeros((nt,len(u)))      
    #copy the initial u array into each row of our new array
    u_n = u.copy()              
    
    #setup some temporary arrays
    flux = numpy.zeros_like(u)
    u_star = numpy.zeros_like(u)

    for t in range(1,nt):
               
        sigma = minmod(u,dx) #calculate minmod slope

        #reconstruct values at cell boundaries
        u_left = u + sigma*dx/2.
        u_right = u - sigma*dx/2.     
        
        flux_left = computeF(u_left) 
        flux_right = computeF(u_right)
        
        #flux i = i + 1/2
        #Russonov flux
        flux[:-1] = 0.5 * (flux_right[1:] + flux_left[:-1] - dx/dt *\
                          (u_right[1:] - u_left[:-1] ))
        
        #rk2 step 1
        u_star[1:-1] = u[1:-1] + dt/dx * (flux[:-2] - flux[1:-1])
        
        u_star[0] = u[0]
        u_star[-1] = u[-1]
        
        
        sigma = minmod(u_star,dx) #calculate minmod slope
    
        #reconstruct values at cell boundaries
        u_left = u_star + sigma*dx/2.
        u_right = u_star - sigma*dx/2.
        
        flux_left = computeF(u_left) 
        flux_right = computeF(u_right)
        
        flux[:-1] = 0.5 * (flux_right[1:] + flux_left[:-1] - dx/dt *\
                          (u_right[1:] - u_left[:-1] ))
        
        u_n[1:-1] = .5 * (u[1:-1] + u_star[1:-1] + dt/dx * (flux[:-2] - flux[1:-1]))
        
        u_n[0] = u[0]
        u_n[-1] = u[-1]
        u = u_n.copy()
        
    return u_n
            
#Basic initial condition parameters
#defining grid size, time steps, CFL condition, u_netc...
nx = 81
dx = .25
dt = .0002  
nt = int(.01/dt + 1)
gamma = 1.4

x = numpy.linspace(-10,10,nx)

initial = initial(nx)
u = computeU(initial)
u_n = muscl(u, nt, dt, dx)
final = final(u_n)

pyplot.figure(figsize=(15,5))
pyplot.subplot(1,3,1)
pyplot.plot(x,final[:,1])
pyplot.title('Velocity')
pyplot.subplot(1,3,2)
pyplot.plot(x,final[:,2])
pyplot.title('Pressure')
pyplot.subplot(1,3,3)
pyplot.plot(x,final[:,0])
pyplot.title('Density')
    
pyplot.show()
