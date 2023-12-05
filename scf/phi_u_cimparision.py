import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import interpolate

def fun(x:float, u:float = 0.5, chi:float = 0.0) -> float:
    return -np.log(1-x)- 2*chi*x - u

def newton_raphson(fun, x, h=1e-8):
    """ Newton-Raphson's method for the argument iteration """
    warnings.filterwarnings('ignore') # to suppress warnings
    try:
        return  x - 2.0 * h * fun(x) / (fun(x+h) - fun(x-h))
    except ZeroDivisionError:
        return  x - 2.0 * h 

def zero_of_function(fun, x0=0.5, precision=10):
    """ finding the zero of function f(x) closest to x0 """ 
    h = 10.0**(-precision)
    x = 1.0
    x_new = x0
    while abs(x_new - x) > h:
        x = x_new
        x_new = newton_raphson(fun, x)
    return round(x_new, precision)  

def phi(u):
    return [zero_of_function(lambda x: fun(x, u=i)) for i in u]

z = np.linspace(0, 40, 41)
N = 100
H = 20
u = 3/8 * np.pi**2 / N**2 * (H**2 - z**2)
interp_func = interpolate.interp1d(z, phi(u))
print(zero_of_function(interp_func, x0 = 1.0))
# print(zero_of_function(var_phi))
# print(integrate.quad(var_phi(u), 0, zero_of_function(lambda x: var_phi(x))))

plt.plot(z, u, label = "u")
plt.plot(z, phi(u), label = "phi")
plt.ylim(0, max(u)+0.01)
plt.legend()
plt.show()