import numpy 
from matplotlib import pyplot
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

#as rho_in decreases shock velocity decreases
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
    i[:mid,] = numpy.array([1, 0., 100.]) 
    i[mid:,] = numpy.array([0.125, 0., 10.])
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
    u = numpy.array([i1, i1 * i2, i3/(gamma-1.)]).T
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
    f = numpy.array([u2, u2**2/u1 + (gamma-1)*(u3-.5*(u2**2/u1)), (u3 + (gamma-1)*(u3-.5*(u2**2/u1))*(u2/u1))]).T
    return f
    
def richtmyer(u, nt, dt, dx):
    """ Computes the solution with Richtmyer scheme
    
    Parameters
    ----------
    u    : array of arrays
            u vector at current time-step
    nt     : int
            Number of time steps
    dt     : float
            Time-step size
    dx     : float
            Mesh spacing
    
    Returns
    -------
    u_n : array of arrays of arrays
            u vector after nt time steps at every point x
    """
    
    u_n = u.copy()
    u_plus = u.copy()
    
    for t in range(1,int(nt)):
        F = computeF(u)
        u_plus[1:] = .5*(u[1:] + u[:-1] - dt/dx * (F[1:]-F[:-1]))
        Fplus = computeF(u_plus)
        u_n[:-1] = u[:-1]- dt/dx * (Fplus[1:] - Fplus[:-1])
        u = u_n.copy()
        
    return u_n
            
#Basic initial condition parameters
#defining grid size, time steps, CFL condition, u_netc...
nx = 81
dx = .25
dt = .0002  
nt = .01/dt + 1 
gamma = 1.4

x = numpy.linspace(-10,10,nx)

i = initial(nx)
u = computeU(i)

u_n = richtmyer(u, nt, dt, dx)