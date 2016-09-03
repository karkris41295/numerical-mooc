'''
Proof that the Runge-Kutta method error is of order 5
Problem taken from Strogatz Ex. 2.8.9
'''

from sympy import *
init_printing()

t, h = symbols('t h') #defining independent variables t and dt
x = Function('x') #defining function x(t)

k1 = diff(x(t),t)*h
k2 = diff(x(t) + .5 * k1, t) * h
k3 = diff(x(t) + .5 * k2, t) * h
k4 = diff(x(t) + k3, t) * h

rka = x(t) + (k1 + 2*k2 + 2*k3 + k4)/6. #runge kutta approximation of x(t+h)
expr1 = series(x(t+h), h) #expr1 stores the taylor series expansion of x(t+h)
rka = rka.expand()
expr2 = expr1 - rka # we see that all terms but h**5 cancel out. Hence, order 5
expr2 = expr2.evalf()

#o = Order(expr2)
'''
OR 

eq1 = Eq(x(t+h), rka)
eq1 = eq1.expand()
eq2 = Eq(x(t+h), expr1)
eq3 = Eq(eq1.lhs-eq2.lhs, eq1.rhs-eq2.rhs)
'''