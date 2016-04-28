import matplotlib.pyplot as plt
import numpy as np

T = 100 #initial time
dt = 0.02 #time step 
t = np.arange(0.0, 100, 0.02) #time grid
N = len(t)

z0 = 100 #altitude
b0 = 10 # upward velocity resulting from gust
zt = 100
g = 9.81

u = np.array([z0,b0])

#initialize array wit z (elevation) values (height down from potential)
z = np.zeros(N)

#Euler's Method 
for i in range(1, N):
    u = u + dt*np.array([u[1], g*(1-u[0]/zt)])
    z[i] = u[0]
    
