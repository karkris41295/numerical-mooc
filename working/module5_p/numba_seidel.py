import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

def p_analytical(x, y):
    X, Y = numpy.meshgrid(x,y)
    
    p_an = numpy.sinh(1.5*numpy.pi*Y / x[-1]) /\
    (numpy.sinh(1.5*numpy.pi*y[-1]/x[-1]))*numpy.sin(1.5*numpy.pi*X/x[-1])
    
    return p_an

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
    fig = pyplot.figure(figsize=(11,7), dpi=100)
    ax = fig.gca(projection='3d')
    X,Y = numpy.meshgrid(x,y)
    surf = ax.plot_surface(X,Y,p[:], rstride=1, cstride=1, cmap=cm.viridis,
            linewidth=0, antialiased=False)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.view_init(30,45)
    
def L2_rel_error(p, pn):
    return numpy.sqrt(numpy.sum((p - pn)**2)/numpy.sum(pn**2))

nx = 128
ny = 128

L = 5
H = 5

x = numpy.linspace(0,L,nx)
y = numpy.linspace(0,H,ny)

dx = L/(nx-1)
dy = H/(ny-1)

p0 = numpy.zeros((ny, nx))

p0[-1,:] = numpy.sin(1.5*numpy.pi*x/x[-1])

from numba import jit
@jit(nopython=True)
def laplace2d_jacobi(p, pn, l2_target):
    '''Solves the Laplace equation using the Jacobi method 
    with a 5-point stencil
    
    Parameters:
    ----------
    p: 2D array of float
        Initial potential distribution
    pn: 2D array of float
        Allocated array for previous potential distribution
    l2_target: float
        Stopping criterion
        
    Returns:
    -------
    p: 2D array of float
        Potential distribution after relaxation
    '''
        
    iterations = 0
    iter_diff = l2_target+1 #init iter_diff to be larger than l2_target
    denominator = 0.0
    ny, nx = p.shape
    l2_diff = numpy.zeros(20000)
    
    while iter_diff > l2_target:
        for j in range(ny):
            for i in range(nx):
                pn[j,i] = p[j,i]
                
        iter_diff = 0.0
        denominator = 0.0
        
        for j in range(1,ny-1):
            for i in range(1,nx-1):
                p[j,i] = .25 * (pn[j,i-1] + pn[j,i+1] + pn[j-1,i] + pn[j+1,i])
                
        
        #Neumann 2nd-order BC
        for j in range(1,ny-1):
            p[j,-1] = .25 * (2*pn[j,-2] + pn[j+1,-1] + pn[j-1, -1])
            
            
        for j in range(ny):
            for i in range(nx):
                iter_diff += (p[j,i] - pn[j,i])**2
                denominator += (pn[j,i]*pn[j,i])
        
        
        iter_diff /= denominator
        iter_diff = iter_diff**0.5
        l2_diff[iterations] = iter_diff
        iterations += 1    
        
    return p, iterations, l2_diff

@jit(nopython=True)
def laplace2d_SOR(p, pn, l2_target, omega):
    '''Solves the Laplace equation using SOR with a 5-point stencil
    
    Parameters:
    ----------
    p: 2D array of float
        Initial potential distribution
    pn: 2D array of float
        Allocated array for previous potential distribution
    l2_target: float
        Stopping criterion
    omega: float
        Relaxation parameter
        
    Returns:
    -------
    p: 2D array of float
        Potential distribution after relaxation
    '''    
    
    iterations = 0
    iter_diff = l2_target + 1 #initialize iter_diff to be larger than l2_target
    denominator = 0.0
    ny, nx = p.shape
    l2_diff = numpy.zeros(20000)
    
    while iter_diff > l2_target:
        for j in range(ny):
            for i in range(nx):
                pn[j,i] = p[j,i]
                
        iter_diff = 0.0
        denominator = 0.0
        
        for j in range(1,ny-1):
            for i in range(1,nx-1):
                p[j,i] = (1-omega)*p[j,i] + omega*.25 * (p[j,i-1] + p[j,i+1] + p[j-1,i] + p[j+1,i])
        
        #Neumann 2nd-order BC
        for j in range(1,ny-1):
            p[j,-1] = .25 * (2*p[j,-2] + p[j+1,-1] + p[j-1, -1])
            
        for j in range(ny):
            for i in range(nx):
                iter_diff += (p[j,i] - pn[j,i])**2
                denominator += (pn[j,i]*pn[j,i])
        
        
        iter_diff /= denominator
        iter_diff = iter_diff**0.5
        l2_diff[iterations] = iter_diff
        iterations += 1
        
    
    return p, iterations, l2_diff

l2_target = 1e-8
omega = 2./(1 + numpy.pi/nx)
p, iterations, l2_diffSORopt = laplace2d_SOR(p0.copy(), p0.copy(), l2_target, 2)

print("Numba Gauss-Seidel method took {} iterations at tolerance {}".format(iterations, l2_target))