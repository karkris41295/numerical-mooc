import numpy

def ftcs(U, V, Du, Dv, F, k, nt, dt, dh):
    
    Uf = numpy.zeros((nt,len(U),len(U)))
    Uf[0,:,:] = U.copy()
    
    for n in range(1, nt):
        Un = U.copy()
        Vn = V.copy()
        
        U[1:-1,1:-1] = Un[1:-1,1:-1] + Du *\
            (dt/dh**2 * (Un[2:,1:-1] - 2*Un[1:-1,1:-1] + Un[:-2,1:-1]) +\
             dt/dh**2 * (Un[1:-1,2:] - 2*Un[1:-1,1:-1] + Un[1:-1,:-2]))-\
             Un[1:-1, 1:-1]*(Vn[1:-1, 1:-1])**2 + F*(1.-Un[1:-1, 1:-1])
        
        V[1:-1,1:-1] = Vn[1:-1,1:-1] + Dv *\
            (dt/dh**2 * (Vn[2:,1:-1] - 2*Vn[1:-1,1:-1] + Vn[:-2,1:-1]) +\
             dt/dh**2 * (Vn[1:-1,2:] - 2*Vn[1:-1,1:-1] + Vn[1:-1,:-2]))+\
             Un[1:-1, 1:-1]*(Vn[1:-1, 1:-1])**2 - (F + k)*Vn[1:-1, 1:-1]
        
  
        # Enforce Neumann BCs
        U[-1,:] = U[-2,:]
        U[:,-1] = U[:,-2]
        U[0,:] = U[1,:]
        U[:,0] = U[:,1]
        V[-1,:] = V[-2,:]
        V[:,-1] = V[:,-2]
        V[0,:] = V[1,:]
        V[:,0] = V[:,1]
        
        Uf[n,:,:] = U.copy()
        
    return Uf
   
X = 5.
Y = 5.

n = 192
#Du, Dv, F, k = 0.00016, 0.00008, 0.035, 0.065 # Bacteria 1 
#Du, Dv, F, k = 0.00014, 0.00006, 0.035, 0.065 # Bacteria 2
#Du, Dv, F, k = 0.00016, 0.00008, 0.060, 0.062 # Coral
#Du, Dv, F, k = 0.00019, 0.00005, 0.060, 0.062 # Fingerprint
#Du, Dv, F, k = 0.00010, 0.00010, 0.018, 0.050 # Spirals
#Du, Dv, F, k = 0.00012, 0.00008, 0.020, 0.050 # Spirals Dense
Du, Dv, F, k = 0.00010, 0.00016, 0.020, 0.050 # Spirals Fast
#Du, Dv, F, k = 0.00016, 0.00008, 0.020, 0.055 # Unstable
#Du, Dv, F, k = 0.00016, 0.00008, 0.050, 0.065 # Worms 1
#Du, Dv, F, k = 0.00016, 0.00008, 0.054, 0.063 # Worms 2
#Du, Dv, F, k = 0.00016, 0.00008, 0.035, 0.060 # Zebrafish

dh = X/(n-1)
T = 80
dt = .9 * dh**2 / (4*max(Du,Dv))
nt = int(T/dt)

x = numpy.linspace(0,X,n)
y = numpy.linspace(0,Y,n)

uvinitial = numpy.load('./uvinitial.npz')
U = uvinitial['U']
V = uvinitial['V']

Uf= ftcs(U.copy(), V.copy(), Du, Dv, F, k, T, dt, dh) # Retrieveing 3-D array of 8000 frames of the concentrations U and V