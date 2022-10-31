""" This is a class of hilbert matrices
"""

from scipy import linalg as lg
from scipy.linalg import hilbert
from scipy.linalg import invhilbert
from scipy.linalg import lu
import numpy as np


class Hilbert():
    """ This class creates a hilbert matrix of a given dimension and provides a
    method to calcute and return its inverse

    :ivar m: Dimension of the matrix
    :vartype m: integer
    """
    def __init__(self, m):
        """ The constructor of the class. Raises a ValueError Exception if the
        dimension is non-integer or is negative.
        """
        self.m = m
        if (not isinstance(m, int)) or m <= 0:
            raise ValueError("Wrong value for dimension of the matrix")
        self.value = hilbert(self.m)
        self.inv = invhilbert(self.m)

    def condition(self):
        """ This method calculates the condition number of the sparse matrix
        
        :return type: float
        """
        return lg.norm(self.value, np.inf)*lg.norm(self.inv, np.inf)

    def LU(self):
        """ This method calculates the LU-decomposition of a fully-populated
        matrix in the same format.

        :return: Lower and Upper triangular matrices 
        :return type: np.ndarray
        """
        [P, L, U] = lu(self.value)
        return P, L, U

    def LU_solve(self, b):
        """
        This method  solves the system of linear equation for a fully populated
        matrix. The algorithm taken to solve the system of linear equation is
        following:

            | P*A = L*U
            | P*A*x = L*U*x
            | L*U*x = P*b
            | L*z = P*b
            | U*x = z

        :param b: the right hand side of equation Ax=b
        :param type: np.array
        :return: the solved equation for x 
        :return type: np.ndarray

        """
        size = np.shape(b)
        if size[0] != self.m:
            raise ValueError("The Matrix b must be an mx1 matrix")
        P, L, U = self.LU()
        b = P.dot(b)
        z = lg.solve(L, b)
        x = lg.solve(U, z)
        return x

def main():
    """ Main function as tests
    """
    n = 4
    i = 0
    matrix = Hilbert(n)
    print(matrix.value)
    print(matrix.inv)
    inv = matrix.inv
    e = np.zeros(n)
    e2 = np.zeros(n)
    e[i] = 1
    e2[i] = 1 + 10**(-8)
    exact = inv[:, i]
    u = matrix.LU_solve(e)
#    u2 = matrix.LU_solve(e2)
#    print(u2)
    print(u)
    print(inv[:, i])
    print(matrix.condition())

if __name__ == '__main__':
    main()
