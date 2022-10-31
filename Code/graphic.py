""" This Module provides function to plot different aspects of Laplace Equation"""


import matplotlib
#matplotlib.use('TkAgg')

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plot_disc_fct as pdf
import Laplace as lp

def graphic_dict(u_hat, n, axis, title=""):
    """ This function plots a vector as discrete function on the unit square

        :param u_hat: The function values on the grid
        :param type: numpy.ndarray
        :param n: dimension of the grid
        :param type: int
        :param axis: axis object of the plot
        :param type: axis Object
        :param title: The title of the plot. Default is empty string
        :param type: string

        :return surf: The surface object
            for further completion in the future (such as setting the colorbar)
        :return type: mpl_toolkits.mplot3d.art3d.Poly3DCollection
    """
    if n < 2:
        raise ValueError("Dimension must be greater than 2")
    if not isinstance(u_hat, np.ndarray):
        raise TypeError("The input must be a numpy array")
    if len(u_hat) != (n-1)**2:
        raise ValueError("The input vector must be of size ((n-1)^d, 1)")

    u_hat_in = u_hat.tolist()
#    u_hat_in = [ elem for lst in u_hat_in for elem in lst]
    surf = pdf.plot_disc_fct(u_hat_in, n, axis, title)
    return surf

def graphic_cont(u, n, axis):
    """ This function plots a continuous function on the unit square

        :param u: The function
        :param type: callable
        :param n: dimension of the meshgrid
        :param type: int
        :param axis: axis object of the plot
        :param type: axis Object

        :return surf: The surface object
            for further completion in the future (such as setting the colorbar)
        :return type: mpl_toolkits.mplot3d.art3d.Poly3DCollection
    """
    if not callable(u):
        raise TypeError("The function must be callable object")
    x = np.arange(0, 1, 1/n)
    y = np.arange(0, 1, 1/n)
    X, Y = np.meshgrid(x, y)
    Z = u(X, Y)
    surf = axis.plot_surface(X, Y, Z, rstride=1, cstride=1,
                             cmap=cm.RdBu, linewidth=0, antialiased=False)
    return surf

def graphic_error(func, exact, n, axis):
    """ In this function plots the error of the solution of the Laplace Equation

        :param func: the numerical solution of the Laplace Equation
        :param type: numpy.ndarray
        :param n: Dimension
        :param type: int
        :param exact: The reference solution
        :param type: numpy.ndarray
        :param axis: Axis Object of the plot
        :param type: axis Object

        :return surf: The surface object
            for further completion in the future (such as setting the colorbar)
        :return type: mpl_toolkits.mplot3d.art3d.Poly3DCollection
    """
    if n < 2:
        raise ValueError("Dimension must be greater than 2")
    if not isinstance(func, np.ndarray) and not isinstance(func, np.ndarray):
        raise TypeError("The input must be a numpy array")
    if len(func) != (n-1)**2 and len(exact) != (n-1)**2:
        raise ValueError("The input vectors must be of size ((n-1)^d, 1)")
    err = abs(func - exact)
    surf = graphic_dict(err, n, axis)
    return surf

def main():
    """ The Main Function"""
    def z_func(x, y):
        return (1-(x**2+y**3))*np.exp(-(x**2+y**2)/2)

    def u(*x):
        """ This is the exact solution of Laplace Equation"""
        u = 1
        for i, j in enumerate(x):
            u = u*j*(1-j)
        return u

    def f_exact(*x):
        """ This function calculates the needed right hand side of Laplace Equation
        for the given exact solution u(x)
        """
        summ = 0
        for i, j in enumerate(x):
            u_temp = 1
            for k in [r for s, r in enumerate(x) if s != i]:
                u_temp = u_temp*k*(1-k)
            summ = summ + 2*u_temp
        return summ

    n = 11
    d = 2
    lap_obj = lp.Laplace(f_exact, d, n, u)
    b = lap_obj.create_b()
    u_hat, u_hat_array = lap_obj.solution(b, 1)
    exact = lap_obj.exact_sol()
    lap_obj = lp.Laplace(f_exact, d, n, u)
    b = lap_obj.create_b()
    u_hat, u_hat_array = lap_obj.solution(b, 1)
    exact = lap_obj.exact_sol()
    fig = plt.figure()
    axis = Axes3D(fig)
    surf = graphic_dict(u_hat, n, axis)
    plt.title("The numerical solution as a discrete function")
    fig.colorbar(surf, shrink=0.5)
    fig = plt.figure()
    axis = Axes3D(fig)
    surf = graphic_cont(z_func, n, axis)
    plt.title("A continuous function z_func as a test function")
    fig.colorbar(surf, shrink=0.5)
    fig = plt.figure()
    axis = Axes3D(fig)
    surf = graphic_cont(u, n, axis)
    plt.title("The exact solution")
    fig.colorbar(surf, shrink=0.5)
    plt.show()
#    fig = plt.figure()
#    axis = Axes3D(fig)
#    surf = graphic_error(u_hat, exact, n, axis)
#    plt.title("The error of the solution as a discrete function")
#    fig.colorbar(surf, shrink=0.5)
#    plt.show()

if __name__ == '__main__':
    main()
