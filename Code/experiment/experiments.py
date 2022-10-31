""" Experiments for solving system of linear equations"""

import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plot_disc_fct as pdf
import numpy as np
import Laplace as lp
import condition_plots as cp
import graphic as gr
import Matrix as M
import Hilbert as H

plt.close('all')

def u(*x):
    """ This is the exact solution of Laplace Equation"""
    u = 1
    for i, j in enumerate(x):
        u = u*np.sin(np.pi*j)
    return u

def f_exact(*x):
    """ This function calculates the needed right hand side of Laplace Equation
    for the given exact solution u(x)
    """
    u = 1
    for i, j in enumerate(x):
        u = u*np.sin(np.pi*x[i])
    return ((np.pi)**2)*len(x)*u


def solution_band(m, cg=1):
    """ calculates and prints the solution of Laplace Equation for different
    dimensions and discretization

        :param m: Dimension range
        :param type: int
        :param cg: indicator if the solution must use the CG or LU methods
        :param type: int
    """
    n = np.arange(2, m)
    for d in range(1, 4):
        for i, j in enumerate(n):
            Lap_obj = lp.Laplace(f_exact, d, j, u)
            b = Lap_obj.create_b()
            lap, lap_array = Lap_obj.solution(b, cg)
            print("the solution of the laplace equation for n = {}".format(j)+
                  " and d = {}".format(d) + " is: \n {}".format(lap))
    return

def error_band_plot(cg=1, tri=1):
    """ This function plots the the error for different dimensions and also
    presents 3D plots for the case d = 2 of the exact and numerical solution of
    the equation as well as the error function.

    Inputs:
        tri(int): indicator if spsolve_triangular or spsolve must be used
            1 for spsolve_triangular, else for spsolve
    """
    for d in range(1, 4):
        cp.err_band(f_exact, u, d, cg)
        if d == 2:
            for n in [4, 11]:
                lap_obj = lp.Laplace(f_exact, d, n, u)
                b = lap_obj.create_b()
                u_hat, u_hat_array = lap_obj.solution(b, cg, tri)
                exact = lap_obj.exact_sol()
                errfun = abs(u_hat - exact)
                fig = plt.figure()
                axis = Axes3D(fig)
                surf = gr.graphic_dict(u_hat, n, axis, "Numerical solution for "+
                                       "n = {}".format(n) + " and d = 2")
                fig.colorbar(surf, shrink=0.5)

                fig = plt.figure()
                axis = Axes3D(fig)
                surf = gr.graphic_dict(exact, n, axis, "Exact solution for "+
                                       "n = {}".format(n) + " and d = 2")
                fig.colorbar(surf, shrink=0.5)

                fig = plt.figure()
                axis = Axes3D(fig)
                surf = gr.graphic_dict(errfun, n, axis, "Error function for "+
                                       "n = {}".format(n) + " and d = 2")
                fig.colorbar(surf, shrink=0.5)
                plt.show()
    return

def sparsity(n):
    """ This method calls the nonzero_band_plot(n) for the m as range

    Inputs:
        n(int): dimension range
    """
    return cp.nonzero_band_plot(n)

def band_cond(n):
    """ This method calls the cond_band_plot(n) for the m as range

    Inputs:
        n(int): dimension range
    """
    return cp.cond_band_plot(n)

def hilbert_sol():
    """ This function calculates the hilbert solution for the fixed given
    data and dimension and plots the error
    No Input is required
    """
    m = [1, 2, 4, 8, 16, 32, 64, 128]
    Err = np.zeros(len(m))
    for k, j in enumerate(m):
        matrix = H.Hilbert(j)
        err = np.zeros(j)
        for i in range(j):
            e = np.zeros(j)
            e[i] = 1.0
            exact = matrix.inv
            exact = exact[:, i]
            err1, err2 = cp.error_build_hilbert(exact, e, matrix)
            err[i] = err1
        Err[k] = max(err)
    plt.figure()
    axis = plt.gca()
    plt.plot(m, Err)
    axis.set_title("Numerical error of the solution\n for a hilbert matrix vs Dimension")
    axis.set_xlabel("dimension m")
    axis.set_ylabel("Numerical error")
    axis.set_xscale("log")
    axis.set_yscale("log")
    plt.show()
    return

def hilbert_cond(m):
    """ This method calls the cond_hilbert_plot(m) for the m as range

    :param m: dimension range
    :param type: int
    """
    return cp.cond_hilbert_plot(m)

def cg_ex(f_exact, u, d, n, cg=1):
    """ This function plots different types of error for the CG-method"""
    cp.err_band(f_exact, u, d, 1)
    cp.err_cg_iteration(f_exact, u, d, n)
    cp.err_diff_tol(f_exact, u, d, n)
    return

def main():
#    solution_band(10)
    error_band_plot(0, 1)
#    sparsity(12)
#    band_cond(12)
#    hilbert_sol()
#    hilbert_cond(15)
    d = 2
    n = 10
    cg_ex(f_exact, u, d, n, 1)

if __name__ == '__main__':
    main()
