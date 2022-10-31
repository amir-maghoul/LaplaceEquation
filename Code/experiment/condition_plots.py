""" Condition Plots"""

import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy as sc
from mpl_toolkits.mplot3d import Axes3D
import Matrix as M
import Laplace as lp
import Hilbert as H
import plot_disc_fct as pdf

plt.close("all")

def cond_band_plot(n):
    """ Plots the condition of a band matrix with respect to its dimension.
 
        :param n: dimension range
        :param type: int
            
    """
    n = np.arange(2, n)
    for d in range(1, 4):
#        if d == 1:
#            n = [2, 11, 101, 1001, 10001]
#        if d == 2:
#            n = [2, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
#        if d == 3:
#            n = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        cond = np.zeros(len(n))
        m = np.zeros(len(n))
        for j in range(2, len(n)+2):
            m[j-2] = (j-1)**d
            matrix = M.Matrix(d, j)
            cond[j-2] = matrix.condition()
        plt.figure()
        axis = plt.gca()
        plt.plot(m, cond)
        axis.set_title("Condition number of a band matrix vs Dimension" +
                       " for d = {}".format(d))
        axis.set_xlabel("dimension m")
        axis.set_ylabel("Condition number")
        axis.set_xscale("log")
        axis.set_yscale("log")
        plt.show()
    return

def cond_hilbert_plot(n):
    """ Plots the condition of a hilbert matrix with respect to its dimension.
        
        :param n: dimension range
        :param type: int
    """
    m = np.arange(1, n)
    cond = np.zeros(len(m))
    for j in range(1, len(m)+1):
        matrix = H.Hilbert(j)
        cond[j-1] = matrix.condition()
    plt.figure()
    axis = plt.gca()
    plt.plot(m, cond)
    axis.set_title("Condition number of a hilbert matrix vs Dimension")
    axis.set_xlabel("dimension m")
    axis.set_ylabel("Condition number")
    axis.set_xscale("log")
    axis.set_yscale("log")
    plt.show()
    return

def nonzero_band_plot(n):
    """ Plots the number of nonzero elements of a band matrix with
    respect to its dimension. Also it plots the number of nonzero elements
    of the LU decomposion of a band matrix with respect to its dimension.

      :param n: dimension range
      :param type: int
    """
    n = np.arange(2, n)
    for d in range(1, 4):
#        if d == 1:
#            n = [2, 11, 101, 1001, 10001]
#        if d == 2:
#            n = [2, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
#        if d == 3:
#            n = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        non_zeros = np.zeros(len(n))
        lu_nonzeros = np.zeros(len(n))
        m = np.zeros(len(n))
        for j in range(2, len(n)+2):
            m[j-2] = (j-1)**d
            matrix = M.Matrix(d, j)
            non_zeros[j-2] = matrix.nonzero_elements()
            lu_nonzeros[j-2] = matrix.LU_nonzeros()
        plt.figure()
        axis = plt.gca()
        plt.plot(m, non_zeros)
        axis.set_title("Number of Nonzero Elements vs Dimension\n" +
                       " for d = {}".format(d))
        axis.set_xlabel("dimension m")
        axis.set_ylabel("number of nonzero elements")
        axis.set_xscale("log")
        axis.set_yscale("log")
        plt.show()

        plt.figure()
        axis = plt.gca()
        plt.plot(m, lu_nonzeros)
        axis.set_title("Number of LU Decomp. Nonzero Elements vs Dimension\n" +
                       " for d = {}".format(d))
        axis.set_xlabel("dimension m")
        axis.set_ylabel("number of nonzero elements")
        axis.set_xscale("log")
        axis.set_yscale("log")
        plt.show()
    return

def err_band(fkt, exact, d, cg = 1, tol = 10**(-8)):
    """ Plots the solution error of a band matrix with respect to its dimension.
    
        :param fkt: the right hand side of Laplace Equation
        :param type: callable
        :param exact: Exact solution of Laplace Equation
        :param type: callable
        :param d: dimension of the field
        :param type: int
        :param cg: indicator if the solution must use the CG or LU methods
        :param type: int
    """
    n = np.arange(2, 11)
    if d == 1:
        n = [2, 11, 101, 1001, 10001]
    if d == 2:
        n = [2, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
    if d == 3:
        n = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    err = np.zeros(len(n))
    m = np.zeros(len(n))
    for i, j in enumerate(n):
        Lap_obj = lp.Laplace(fkt, d, j, exact)
        b = Lap_obj.create_b()
        err[i] = Lap_obj.sol_err(b, cg, tol)
        m[i] = (j-1)**d
    plt.figure()
    axis = plt.gca()
    axis.plot(m, err, '-o')
    if cg == 1:
        axis.set_title("Numerical error of the solution \n"+
                       " using conjugate gradient" +
                       " for d = {}".format(d))
    else:
            axis.set_title("Numerical error of the solution \n"+
                   " using LU-decomposition" +
                   " for d = {}".format(d))
    axis.set_xlabel("dimension m")
    axis.set_ylabel("Numerical error")
    axis.set_xscale("log")
    axis.set_yscale("log")
    plt.show()
    return

#TODO:
    
def err_cg_iteration(fkt, exact, d, n, tol = 10**(-8)):
    """ Plots the solution error of a band matrix with respect to its dimension.

    :param fkt: the right hand side of Laplace Equation
    :param type: callable
    :param exact: Exact solution of Laplace Equation
    :param type: callable
    :param d: Dimension of the field
    :param type: int
    :param n: Fineness of discritization
    :param type: int
    :param tol: Tolerance of the CG method
    :param type: float
    """
    plt.figure()
    axis = plt.gca()
    n = np.arange(2, n)
    for i, j in enumerate(n):
        Lap_obj = lp.Laplace(fkt, d, j, exact)
        b = Lap_obj.create_b()
        norm = Lap_obj.sol_err_cg_iter(b, tol)
        n = np.arange(len(norm))
        axis.plot(n, norm, '-o', label = "n ={}".format(j)+", d ={}".format(d))
        axis.set_title("The development of the error of the solution"+
                       "\n for CG method at each iteration for d = {}".format(d))
        axis.set_xlabel("i-th Iteration")
        axis.set_ylabel("Solution Error")
    #    axis.set_xscale("log")
        axis.set_yscale("log")
        axis.legend()
    plt.show()    
    return

def err_diff_tol(fkt, exact, d, n):
    """This function plots the solution error for different values of a 
    tolerance in the CG method for a given dimension

    :param fkt: the right hand side of Laplace Equation
    :param type: callable
    :param exact: Exact solution of Laplace Equation
    :param type: callable
    :param d: Dimension
    :param type: int
    :param n: Fineness of discritization
    :param type: int
    """
    h = 1/n
    tol = [h**(-2), 1, h**2, h**4, h**6]
    tol = sorted(tol)
    err = np.zeros(len(tol))
    plt.figure()
    axis = plt.gca()
    for i, j in enumerate(tol):
        Lap_obj = lp.Laplace(fkt, d, n, exact)
        b = Lap_obj.create_b()
        err[i] = Lap_obj.sol_err(b, 1, j)
    axis.plot(tol, err, '-o')
    axis.set_title("Solution error for for CG method"+
                   "\n for n ={}".format(n)+" and d ={}".format(d))
    axis.set_xlabel("tol")
    axis.set_ylabel("Solution Error")
    axis.set_xscale("log")
#    axis.set_yscale("log")
    axis.legend()
    plt.show()
    return

def error_build_hilbert(exact, b, matrix=None):
    """ This is an auxilary function to create the error arrays for a given
    reference solution and a right hand side b. 

    :param exact: the reference solution as vector
    :param type: np.ndarray
    :param b: Right hand side of the system using a hilbert matrix
    :param type: np.ndarray
        
    :return err1: the error vector
    :return type: np.ndarray
    :return err2: residue vector
    :return type: np.ndarray
    """
    m = len(b)
    if len(exact) != m :
        raise ValueError("the reference solution"+
                         "must be of size {}".format(m))
    if matrix == None:
        matrix = H.Hilbert(m)
    u = matrix.LU_solve(b).T
    err1 = np.linalg.norm((u-exact), np.inf)
    err2 = np.linalg.norm(((matrix.value*u)-(matrix.value*exact)), np.inf)
    return err1, err2
        
def error_hilbert():
    """ Plots the solution error of a hilbert matrix with respect to its dimension.
    No inputs and outputs
    """
    N = np.arange(1, 4)
    err = np.zeros(len(N))
    for j in range(1, len(N)+1):
        b = np.arange(j)
        exact = np.arange(j)
        err1, err2 = error_build_hilbert(exact, b)
        err[j-1] = err1
    plt.figure()
    axis = plt.gca()
    plt.plot(N, err)
    axis.set_title("Numerical error of the solution \n for a hilbert matrix vs Dimension")
    axis.set_xlabel("dimension m")
    axis.set_ylabel("Numerical error")
    plt.show()
    return

def residue_band(fkt, exact, d, cg = 1):
    """ Plots the solution residue of a band matrix with respect to its dimension.
    
        :param fkt: the right hand side of Laplace Equation
        :param type: callable
        :param exact: Exact solution of Laplace Equation
        :param type: callable
        :param cg: indicator if the solution must use the CG or LU methods
        :param type: int
    """ 
    n = np.arange(2, 10)
    res = np.zeros(len(n))
    m = np.zeros(len(n))
    for i, j in enumerate(n):
        Lap_obj = lp.Laplace(fkt, d, j, exact)
        Mat_Obj = M.Matrix(d, j)
        b = Lap_obj.create_b()
        [u, u_array] = Lap_obj.solution(b, cg, tri=1)
        matrix = Mat_Obj.convert()
        Mu = matrix*u
        res[i] = sc.linalg.norm(Mu - b, np.inf)
        m[i] = (j-1)**d
    plt.figure()
    axis = plt.gca()
    plt.plot(m, res)
    axis.set_title("Residue for"+
                    " a band matrix vs Dimension" +
                    " for d = {}".format(d))
    axis.set_xlabel("dimension m")
    axis.set_ylabel("Residue ")
    plt.show()
    return

def residue_hilbert():
    """ Plots the solution error of a hilbert matrix with respect to its dimension.
    No inputs and outputs
    """
    N = np.arange(1, 4)
    err = np.zeros(len(N))
    for j in range(1, len(N)+1):
        b = np.arange(j)
        exact = np.random.rand(j, 1)
        err1, err2 = error_build_hilbert(exact, b)
        err[j-1] = err2
    plt.figure()
    axis = plt.gca()
    axis.plot(N, err)
    axis.set_title("Residual of the solution \n for a hilbert matrix vs Dimension")
    axis.set_xlabel("dimension m")
    axis.set_ylabel("Numerical error")
    plt.show()
    return

def main():
    """the main function as test"""
    def func(x, y):
        return 1
    
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

    d = 2
    n = 15
#    err_band(f_exact, u, d, 0)
#    err_band(f_exact, u, d, 1)
#    residue_band(f_exact, u, 2)
#    error_hilbert()
#    residue_hilbert()
#    nonzero_band_plot(10)
#    cond_hilbert_plot(4)
#    cond_band_plot(10)
#    err_cg_iteration(f_exact, u, d, n)
    err_diff_tol(f_exact, u, d, n)
    
if __name__ == '__main__':
    main()
