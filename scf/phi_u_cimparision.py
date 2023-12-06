import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy import interpolate
from scipy import integrate

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
    x = x0+1.0
    x_new = x0
    while abs(x_new - x) > h:
        x = x_new
        x_new = newton_raphson(fun, x)
    return round(x_new, precision)  

def phi_profile(u):
    return [zero_of_function(lambda x: fun(x, u=i)) for i in u]

def potential(z, N, H, eta=1.0):
    return 3/8 * (np.pi**2 / N**2) * (H**2 - z**2)

def theta_H(H, N = 100):
    z = np.linspace(0, N, 100)
    u = potential(z, N, H)
    phi = phi_profile(u)
    interp_func = interpolate.interp1d(z, phi, fill_value='extrapolate')
    h = fsolve(interp_func, H, xtol = 1e-8)
    return integrate.quad(interp_func , 0, h)[0]

if __name__ == '__main__':
    theta = 10
    N = 100
    h0 = np.sqrt(N)
    
    x = np.linspace(1, N, 100)
    y = lambda x: theta_H(x, N = N)-theta
    print(fsolve(y, 100.0))
    # plt.plot(x, y(x))
    # plt.show()
    
    # func = lambda x: (theta_H(x, N = N)-theta)   #Not working yet. Function returns just h0 itself
    # print(fsolve(func, h0, xtol = 1e-8)) #Not working yet. Function returns just h0 itself







# plt.plot(z, u, label = "u")
# plt.plot(z, phi, label = "phi")
# plt.ylim(0, max(u)+0.01)
# plt.legend()
# plt.show()