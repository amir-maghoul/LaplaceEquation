""" Conjugate Gradient Method """

import numpy as np
import scipy.sparse as sp
from numpy import linalg as LA
from scipy.sparse import linalg as lg

def cg1(A, b, u0 = None, tol = 10**(-8), itmax = None):
    """
    Conjugate Gradient Method to solve Ax=b.

      :param b: The right hand side of equation
      :param type: numpy.ndarray
      :param u0: Initial guess of the solution. Default is the zero matrix.
      :param type: numpy.ndarray
      :param tol: Accuracy tolerance for the algorithm stopping criterion. Default
               1.0e-8.
      :param type: float
      :param itmax: Maximun iterations number allowed. Default value is 1000
      :param type: int
    """
    size = b.shape
    if u0 == None:
        u0 = np.zeros(size)
#    if not isinstance(u0, np.ndarray) or not isinstance(b, np.ndarray):
#        raise ValueError("The initial value x and the right hand side b "+
#                         "must be numpy arrays")
    if not isinstance(u0, np.ndarray):
        u0 = np.asarray(u0)
    if not isinstance(b, np.ndarray):
        b = np.asarray(b)
        
    if itmax == None:
        itmax = 1000
    k = 0
    u = u0
    g = A.dot(u0) - b
    d = -g
    while k < itmax:
        alpha = LA.norm(g, 2)**2/(np.dot(d, (A.dot(d)))) 
        if k == 0:
            u_tmp = u + alpha*d
        else:
            u_tmp = u[-1, :] + alpha*d
        u = np.vstack([u, u_tmp])
#        g_next = g + alpha*(A.dot(d))
        g_next = A.dot(u_tmp) - b
        beta = (LA.norm(g_next, 2)/LA.norm(g, 2))**2
        d = -g_next + beta*d
        if LA.norm(g_next, 2) <= tol*LA.norm(g, 2):
            break
        else:
            g = g_next
            k = k + 1
    return u

def cg2(A, b, u0 = None, tol = 10**(-8), itmax = None):
    """
    Conjugate Gradient Method to solve Ax=b.

      :param A: Symmetric positive definite coefficients matrix.
      :param type: Sparse matrix
      :param b: The right hand side of equation
      :param type: numpy.ndarray
      :param u0: Initial guess of the solution. Default is the zero matrix.
      :param type: numpy.ndarray
      :param tol: Accuracy tolerance for the algorithm stopping criterion. Default
               1.0e-8.
      :param type: float
      :param itmax: Maximun iterations number allowed. Default value is 1000
      :param type: int
    """
    (rows, columns) = A.shape
    if rows != columns:
        raise ValueError("A must be a square matrix")
    if u0 == None:
        u0 = np.zeros((rows, 1))
#    if not isinstance(u0, np.ndarray) or not isinstance(b, np.ndarray):
#        raise ValueError("The initial value x and the right hand side b"+
#                         "must be numpy arrays")
    if not isinstance(u0, np.ndarray):
        u0 = np.asarray(u0)
    if not isinstance(b, np.ndarray):
        b = np.asarray(b)
        
    if itmax == None:
        itmax = 1000
    k = 0
    u = u0
    r_old = b - A.dot(u0)
    if LA.norm(r_old, 2) <= tol:
        return u0
    p = r_old
    while k < itmax:
        alpha = np.sum(r_old.T*r_old)/(np.sum(p.T*(A.dot(p))))
        if k == 0:
            u_tmp = u + alpha*p
        else:
            u_tmp = u[-1, :] + alpha*p
        u = np.vstack([u, u_tmp])
        r_new = r_old - alpha*(A.dot(p))
        if LA.norm(r_new, 2) <= tol:
            break
        beta = np.sum(r_new.T*r_new)/np.sum(r_old.T*r_old)
        p = r_new + beta*p
        r_old = r_new
        k = k + 1
    return u[-1, :]

def main():
#    A = np.array([[ 2., -1.,  0.],
#                  [-1.,  2., -1.],
#                  [ 0., -1.,  2.]])
#    b = np.array([1, 1, 1])
    A = np.array([[ 4., 1.],
                  [1., 3.]])
    b = np.array([1, 2])
    size = b.shape
    x0 = np.zeros(size)
    x0 = np.array([2, 1])
#    x0 = np.array([0, 0, 0])
    x = cg1(A, b, x0)
    print(x)

if __name__ == '__main__':
    main()
