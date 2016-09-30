import numpy
from matplotlib import pyplot
from matplotlib import rcParams, cm
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

def ftcs(T, nt, alpha, dt, dx, dy, q):

    #force j_mid and i_mid to be integers so we can use them as indices
    #for the array T
    j_mid = int((numpy.shape(T)[0])/2) 
    i_mid = int((numpy.shape(T)[1])/2)
   
    for n in range(nt):
        Tn = T.copy()
        T[1:-1,1:-1] = Tn[1:-1,1:-1] + alpha *\
            (dt/dy**2 * (Tn[2:,1:-1] - 2*Tn[1:-1,1:-1] + Tn[:-2,1:-1]) +\
             dt/dx**2 * (Tn[1:-1,2:] - 2*Tn[1:-1,1:-1] + Tn[1:-1,:-2]))
  
        # Enforce Neumann BCs
        T[-1,:] = T[-2,:] + q*dx
        T[:,-1] = T[:,-2] + q*dx
        
        # Check if we reached T=70C
        if T[j_mid, i_mid] >= 70:
            print ("Center of plate reached 70C at time {0:.2f}s.".format(dt*n))
            break
        
    if T[j_mid, i_mid]<70:
        print ("Center has not reached 70C yet, it is only {0:.2f}C.".format(T[j_mid, i_mid]))
        
    return T
    
L = 1.0e-2
H = 1.0e-2

nx = 21
ny = 21
nt = 500

dx = L/(nx-1)
dy = H/(ny-1)

x = numpy.linspace(0,L,nx)
y = numpy.linspace(0,H,ny)

alpha = 1e-4

Ti = numpy.ones((ny, nx))*20
Ti[0,:]= 100
Ti[:,0] = 100
q = 0 # neumann boundary condition

sigma = 0.25
dt = sigma * min(dx, dy)**2 / alpha
T = ftcs(Ti.copy(), nt, alpha, dt, dx, dy, q)

pyplot.figure(figsize=(10,8))
pyplot.contourf(x,y,T,20,cmap=cm.viridis)
pyplot.xlabel('$x$')
pyplot.ylabel('$y$')
pyplot.colorbar();
pyplot.show()