import numpy
import sympy
from matplotlib import pyplot
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

v_max = 80. #km/hr
L = 11. #km
rho_max = 250 #cars/km
nx = 51
dx = L/(nx-1)
dt = .001 #hrs

x = numpy.linspace(0,L,nx)
rho0 = numpy.ones(nx)*10
rho0[10:20] = 50