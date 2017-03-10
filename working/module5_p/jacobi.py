import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plot_3D(x, y, p):
    '''Creates 3D plot with appropriate limits and viewing angle
    
    Parameters:
    ----------
    x: array of float
        nodal coordinates in x
    y: array of float
        nodal coordinates in y
    p: 2D array of float
        calculated potential field
    
    '''
    fig = plt.figure(figsize=(11,7), dpi=100)
    ax = fig.gca(projection='3d')
    X,Y = np.meshgrid(x,y)
    surf = ax.plot_surface(X,Y,p[:], rstride=1, cstride=1, cmap=cm.viridis,
            linewidth=0, antialiased=False)

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.view_init(30,45)
    
def p_analytical(x, y):
    X, Y = np.meshgrid(x,y)
    
    p_an = np.sinh(1.5*np.pi*Y / x[-1]) /\
    (np.sinh(1.5*np.pi*y[-1]/x[-1]))*np.sin(1.5*np.pi*X/x[-1])
    
    return p_an

def laplace2d(p, l2_target):
    '''Iteratively solves the Laplace equation using the Jacobi method
    
    Parameters:
    ----------
    p: 2D array of float
        Initial potential distribution
    l2_target: float
        target for the difference between consecutive solutions
        
    Returns:
    -------
    p: 2D array of float
        Potential distribution after relaxation
    '''
    
    l2norm = 1
    pn = np.empty_like(p)
    while l2norm > l2_target:
        pn = p.copy()
        p[1:-1,1:-1] = .25 * (pn[1:-1,2:] + pn[1:-1, :-2] \
                              + pn[2:, 1:-1] + pn[:-2, 1:-1])
        
        ##2nd order Neumann B.C. along x = L
        p[1:-1, -1] = .25 * (2*pn[1:-1,-2] + pn[2:, -1] + pn[:-2, -1])
        l2norm = L2_error(p, pn)
     
    return p

def L2_error(p, pn):
    return np.sqrt(np.sum((p - pn)**2)/np.sum(pn**2)) #dividing by denominator to ccount for varying numbers of i and j as grid size grows

def laplace_IG(nx):
    '''Generates initial guess for Laplace 2D problem for a 
    given number of grid points (nx) within the domain [0,1]x[0,1]
    
    Parameters:
    ----------
    nx: int
        number of grid points in x (and implicitly y) direction
        
    Returns:
    -------
    p: 2D array of float
        Pressure distribution after relaxation
    x: array of float
        linspace coordinates in x
    y: array of float
        linspace coordinates in y
    '''

    ##initial conditions
    p = np.zeros((nx,nx)) ##create a XxY vector of 0's

    ##plotting aids
    x = np.linspace(0,1,nx)
    y = x

    ##Dirichlet boundary conditions
    p[:,0] = 0
    p[0,:] = 0
    p[-1,:] = np.sin(1.5*np.pi*x/x[-1])
    
    return p, x, y

nx_values = [11, 21, 41, 81]
l2_target = 1e-8

error = np.empty_like(nx_values, dtype=np.float)


for i, nx in enumerate(nx_values):
    p, x, y = laplace_IG(nx)
    
    p = laplace2d(p.copy(), l2_target)
    
    p_an = p_analytical(x, y)
    
    error[i] = L2_error(p, p_an)
    
plt.figure(figsize=(6,6))
plt.grid(True)
plt.xlabel(r'$n_x$', fontsize=18)
plt.ylabel(r'$L_2$-norm of the error', fontsize=18)

plt.loglog(nx_values, error, color='k', ls='--', lw=2, marker='o')
plt.axis('equal');

