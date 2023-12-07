import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, interpolate
from scipy.optimize import fsolve


def fun(x: float, u: float = 0.5, chi: float = 0.0) -> float:
    return -np.log(1 - x) - 2 * chi * x - u


def newton_raphson(fun, x, h=1e-8):
    """Newton-Raphson's method for the argument iteration"""
    warnings.filterwarnings("ignore")  # to suppress warnings
    try:
        return x - 2.0 * h * fun(x) / (fun(x + h) - fun(x - h))
    except ZeroDivisionError:
        return x - 2.0 * h


def zero_of_function(fun, x0=0.5, precision=10):
    """finding the zero of function f(x) closest to x0"""
    h = 10.0 ** (-precision)
    x = x0 + 1.0
    x_new = x0
    while abs(x_new - x) > h:
        x = x_new
        x_new = newton_raphson(fun, x)
    return round(x_new, precision)


def phi_profile(u, chi):
    return [zero_of_function(lambda x: fun(x, u=i, chi=chi)) for i in u]


def potential(z, N, H, eta=1.0):
    return 3 / 8 * eta**2 * (np.pi**2 / N**2) * (H**2 - z**2)


def theta_H(H, N=100, chi: float = 0.0):
    z = np.linspace(0, N, 100)
    u = potential(z, N, H)
    phi = phi_profile(u, chi=chi)
    interp_func = interpolate.interp1d(z, phi, fill_value="extrapolate")
    h = fsolve(interp_func, H, xtol=1e-8)
    return integrate.quad(interp_func, 0, h)[0]


def thickness(N, theta, chi):
    h0 = np.sqrt(N)
    x = np.linspace(1, N, 100)
    y = [
        theta_H(i, N=N, chi=chi) - theta
        for i in x
        if theta_H(i, N=N, chi=chi) == theta_H(i, N=N, chi=chi)
    ]
    x = x[: len(y)]
    y_interp = interpolate.interp1d(x, y, fill_value="extrapolate")
    return fsolve(y_interp, h0, xtol=1e-8)


if __name__ == "__main__":
    theta = 1
    chi = 0.0
    N = 100

    H = thickness(N, theta, chi)
    z = np.linspace(0, 40, 41)
    u = potential(z, N, H)
    phi = phi_profile(u, chi=chi)

    plt.plot(z, phi, label="phi")
    plt.plot(z, u, label="U")
    plt.ylim(0, max(u) + 0.001)
    plt.legend()
    plt.show()
