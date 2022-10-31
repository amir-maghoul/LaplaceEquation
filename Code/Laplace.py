""" This Module provides a class to solve the PDE for the Laplace Operator"""

import numpy as np
import scipy.sparse as sp
import scipy as sc
import Matrix as M



class Laplace():
    """ This is a class which implements methods to solve the Laplace Equation
        It uses a function to evaluate the right hand side of the equation and
        a function or numpy array as the exact solution.

        :ivar fkt: the right hand side of Laplace Equation
        :vartype fkt: callable or numpy.ndarray
        :ivar d: Dimension of the field
        :vartype d: int
        :ivar n: Fineness of discritization
        :vartype n: int
        :ivar exact: The exact solution of Laplace equation
        :vartype exact: callable or numpy.ndarray
    """
    def __init__(self, fkt, d, n, exact=None):
        self.d = d
        self.n = n
        self.func = fkt
        self.exact = exact
        if exact != None:
            self.exact = exact
            if not callable(exact):
                if ((isinstance(exact, np.ndarray) and len(exact) != (n-1)**d)
                        or not isinstance(exact, np.ndarray)):
                    raise ValueError("The exact function must be either"+
                                     "callable or a vector of size (n-1)^d")
        if (not isinstance(self.d, int)) and (not isinstance(self.n, int)):
            raise TypeError("d and m must be of integer type")
        if d < 1 or d > 3:
            raise ValueError("d must be 1, 2 or 3")
        if n < 2:
            raise ValueError("n must not be smaller than 2")
        if not callable(fkt):
            if ((isinstance(fkt, np.ndarray) and len(exact) != (n-1)**d)
                    or not isinstance(exact, np.ndarray)):
                raise ValueError("The function must be either"+
                                 "callable or a vector of size (n-1)^d")

    def create_mesh(self):
        """ This method discretisizes the Domain
        No Inputs

        :returns: X(numpy.ndarray): The discretisized domain as an array
        """
        X = np.zeros(((self.n-1)**self.d, self.d))
        if self.d == 1:
            for i in range(1, self.n):
                X[i-1] = i/self.n
            return X
        if self.d == 2:
            for j in range(1, self.n):
                for i in range(1, self.n):
                    m = i + (self.n-1)*(j-1)
                    X[m-1] = [i/self.n, j/self.n]
            return X
        else:
            for l in range(1, self.n):
                for j in range(1, self.n):
                    for i in range(1, self.n):
                        m = i + (self.n-1)*(j-1) + ((self.n-1)**2)*(l-1)
                        X[m-1] = [i/self.n, j/self.n, l/self.n]
            return X

    def create_coefficient(self):
        """ This method creates the coefficient matrix by calling the Matrix
        class. It also returns the corresponding object of Matrix class for
        future use.
        No Inputs

        :returns: A(Matrix): Object of Matrix class\n
        :returns: A.convert()(csc_sparse): The coefficient matrix in sparse\n
        """
        A = M.Matrix(self.d, self.n)
        return A, A.convert()

    def create_b(self):
        """
        This methods creates the corresponding right hand side of the numerical
        form of the Laplace Equation using the meshgrid created by create_mesh
        No Inputs

        :returns: b(numpy.ndarray): The right hand side of numerical Laplace Equation
        """
        X = self.create_mesh()
        f = np.zeros((len(X), 1))
        if callable(self.func):
            for i in range(len(X)):
                if self.d == 1:
                    f[i] = self.func(X[i])
                elif self.d == 2:
                    f[i] = self.func(X[i, 0], X[i, 1])
                else:
                    f[i] = self.func(X[i, 0], X[i, 1], X[i, 2])
        else:
            return (1/(self.n)**2)*self.func
        b = (1/(self.n)**2)*f
        return b.flatten()

    def solution(self, b, cg=1, tol=10**(-8), tri=1, u0 = None):
        """ This method solves the Ax=b for a given b

        :param b: The given right hand side
        :param type: numpy.ndarray
        :param tri: indicator if spsolve_triangular or spsolve must be used
        :param type: int
        :param tol: Tolerance of the error in the CG method
        :param type: float
        :returns: x(numpy.ndarray): The solution
        """
        if not isinstance(cg, int) and not isinstance(tri, int):
            raise ValueError("cg and tri both must be integers")
        [Obj, A] = self.create_coefficient()
        if cg == 1:
            x_tmp = Obj.cg(b, u0, tol)
            x = np.array(x_tmp[-1, :])
        else:
            x = Obj.LU_solve(b, tri)
            x_tmp = []
        return x, x_tmp

    def exact_sol(self):
        """ This method calculates the exact solution on the grid points. Returns
        the attribute "exact" if it is already a vector on the points.
        No Inputs

        :returns: Exact(numpy.ndarray): the evaluated exact solution on the grid points
        """
        if self.exact is None:
            return None
        if isinstance(self.exact, np.ndarray):
            return self.exact
        else:
            X = self.create_mesh()
            Exact = np.zeros((len(X), 1))
            for i in range(len(X)):
                if self.d == 1:
                    Exact[i] = self.exact(X[i])
                elif self.d == 2:
                    Exact[i] = self.exact(X[i, 0], X[i, 1])
                else:
                    Exact[i] = self.exact(X[i, 0], X[i, 1], X[i, 2])
            return Exact

    def sol_err(self, b, cg=1, tol=10**(-8), u0 = None):
        """ This method calculates the Absolute error of the solution for a given
        right hand side

        :param b: Right hand side of Laplace equation
        :param type: numpy.ndarray
        :param cg: indicator if the solution must use the CG or LU methods
        :param type: int
        :param tol: Tolerance of the error in the CG method
        :param type: float
        :returns: The absolute error
        :return type: float
        """
        x = self.exact_sol()
        u, u_array = self.solution(b, cg, tol, 1, u0)
        if cg == 1:
            return sc.linalg.norm((u - x.T).T, np.inf)
        else:
            return sc.linalg.norm(u - x, np.inf)

    def sol_err_cg_iter(self, b, tol=10**(-8), u0 = None):
        """ This function calculates the error of each iteration of the CG
        method

        :param b: Right hand side of Laplace equation
        :param type: numpy.ndarray
        :param tol: Tolerance of the error in the CG method
        :param type: float
        :returns: numpy.ndarray of the absolute error of each iteration
        :return type: numpy.ndarray
        """
        x = self.exact_sol()
        u, u_array = self.solution(b, 1, tol, 1, u0)
        err_func = (u_array - x.T).T
        iteration = len(u_array)
        norm = np.zeros((iteration, 1))
        for i in range(iteration):
            norm[i] = sc.linalg.norm(err_func[:, i], np.inf)
        return norm.flatten()

def main():
    """The main Function"""

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

    L = Laplace(f_exact, 1, 10, u)
    b = L.create_b()
#    print(L.create_mesh())
#    print(L.create_b())
#    print(L.solution(b, 0)[0])
#    print(L.solution(b, 1)[0])
#    print(L.exact_sol())
    print(L.sol_err(b, 0))
    print(L.sol_err(b, 1))
#    print(L.sol_err_cg_iter(b))

if __name__ == '__main__':
    main()
                      