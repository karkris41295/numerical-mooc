import numpy
from matplotlib import pyplot
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
from scipy.sparse import coo_matrix

L = 1. # 1m rod
nt = 100 # 100 time-steps
nx = 51 #51 points in space
alpha = 1.22e-3  #thermal diffusivity

q = -50. # neumann condition at x = 1, dT/dx = q

dx = L/(nx-1)

qdx = q*dx

Ti = numpy.zeros(nx) # initial conditions
Ti[0] = 100

from scipy.linalg import solve

def generateMatrix(N, sigma):
    """ Computes the matrix for the diffusion equation with backward Euler
        Dirichlet condition at i=0, Neumann at i=-1
    
    Parameters:
    ----------
    T: array of float
        Temperature at current time step
    sigma: float 
        alpha*dt/dx^2
    
    Returns:
    -------
    A: 2D numpy array of float
        Matrix for diffusion equation
    """
   
    # Setup the diagonal
    d = numpy.diag(numpy.ones(N-2)*(2+1./sigma))
    
    # Consider Neumann BC
    d[-1,-1] = 1+1./sigma
    
    # Setup upper diagonal
    ud = numpy.diag(numpy.ones(N-3)*-1, 1)
    
    # Setup lower diagonal
    ld = numpy.diag(numpy.ones(N-3)*-1, -1)
    
    A = d + ud + ld
    
    return A
    
def generateRHS(T, sigma, qdx):
    """ Computes right-hand side of linear system for diffusion equation
        with backward Euler
    
    Parameters:
    ----------
    T: array of float
        Temperature at current time step
    sigma: float
        alpha*dt/dx^2
    qdx: float
        flux at right boundary * dx
    
    Returns:
    -------
    b: array of float
        Right-hand side of diffusion equation with backward Euler
    """
    
    b = T[1:-1]*1./sigma
    # Consider Dirichlet BC
    b[0] += T[0]
    # Consider Neumann BC
    b[-1] += qdx
    
    return b

def implicit_btcs(T, A, nt, sigma, qdx):
    """ Advances diffusion equation in time with implicit central scheme
    This method implies a considerable computational cost.
   
    Parameters:
    ----------
    T: array of float
        initial temperature profile
    A: 2D array of float
        Matrix with discretized diffusion equation
    nt: int
        number of time steps
    sigma: float
        alpha*td/dx^2
        
    qdx: float
        flux at right boundary * dx
    Returns:
    -------
    T: array of floats
        temperature profile after nt time steps
    """
    
    for t in range(nt):
        Tn = T.copy() #trying to be safe I guess
        b = generateRHS(Tn, sigma, qdx)
        # Use numpy.linalg.solve
        T_interior = solve(A,b) # Gives us the Tn+1's
        T[1:-1] = T_interior
        # Enforce Neumann BC (Dirichlet is enforced automatically)
        T[-1] = T[-2] + qdx

    return T
    
sigma = 0.5  # we can use any sigma now and the soluton won't blow up
dt = sigma * dx*dx/alpha 
nt = 1000

A = generateMatrix(nx, sigma)

T = implicit_btcs(Ti.copy(), A, nt, sigma, qdx)
pyplot.plot(numpy.linspace(0,1,nx), T, color='#003366', ls='-', lw=3);
pyplot.xlabel('Length of the rod')
pyplot.ylabel('Temperature')