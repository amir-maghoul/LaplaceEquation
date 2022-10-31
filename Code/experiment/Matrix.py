# -*- coding: utf-8 -*-
"""
This program creates a class of band matrices for d = 1, 2, 3.
"""
import sys
import numpy as np
import scipy.sparse as sp
from numpy import linalg as LA
from scipy.sparse import linalg as lg



class Matrix():
    """
    This is a class of band matrices. This class provides a method to
    create coefficients band matrix as a sparse-Matrix as well as methods to
    count the number of zero and nonzero entries seperately, plus the relative
    number of zero and nonzero entries

    :ivar d: dimension of the field. Value must be to 1, 2 or 3
    :vartype d: integer
    :ivar n: Fineness of discretization. It must be a nonnegative integer
            greater than or equal to 2
    :vartype n: integer

    """

    def __init__(self, d, n):
        """
        The constructor raises TypeError
        exceptions if the d and n are not integers and a ValueError exception
        if d is not equal to 1, 2 or 3. It also raises a ValueError exception
        if n is smaller than 2. Additionally, in order to make the method
        "sparse_matrix" more efficient, a attribute "matrix" is defined.
        """
        self.d = d
        self.n = n
        self.matrix = None
        if (not isinstance(self.d, int)) and (not isinstance(self.n, int)):
            raise TypeError("d and m must be of integer type")

        if d < 1 or d > 3:
            raise ValueError("d must be 1, 2 or 3")
        if n < 2:
            raise ValueError("n must not be smaller than 2")

    def sparse_matrix(self):
        """
        This method returns the object as a sparse matrix with l = k = d
        in the static method "create_sparse_matrix". The method returns the
        attribute "matrix" if its value is not None (the matrix is created
        before) or creates, returns and sets the attribute "matrix" with the
        newly created matrix.

        :param d: dimension of the field. Value must be 1, 2 or 3
        :type d: integer
        :return: The coefficient matrix as a sparse-Matrix
        :rtype: scipy.sparse.dok.dok_matrix
        """
        if self.matrix == None:
            self.matrix = self.create_sparse_matrix(self.d, self.d, self.n)
        return self.matrix


    @staticmethod
    def create_sparse_matrix(l, k, n):
        """
        This method is a recursive method to construct a :math:`(n-1)^{2l}`
        Block-Band sparse Matrix

        The method raises TypeError exception if the l, k and n are not nonzero
        integers.

        :param l: power of the dimension of the matrices. It expands the matrix
                in the previous iteration
        :type l: int
        :param k: coefficient of the diagonal values of the matrix
        :type k: int
        :param n: Fineness of discretization. Here it serves as the base dimension
                of the matrices
        :type n: int
        :return: Block-Band sparse matrix
        :rtype: scipy.sparse.dok.dok_matrix

        """
        size = (n-1)**(l-1)
        step = n-1
        if (not isinstance(l, int)) and \
            (not isinstance(k, int)) and (not isinstance(n, int)):
            raise TypeError("The inputs must be integers")
        if n == 2:
            matrix_big = sp.dok_matrix((1, 1))
            matrix_big[0, 0] = 2*k
            return matrix_big
        if l == 1:
            matrix_big = sp.dok_matrix((n-1, n-1))
            matrix_big[0, 0] = 2*k
            matrix_big[0, 1] = -1
            matrix_big[n-2, n-3] = -1
            matrix_big[n-2, n-2] = 2*k
            for i in range(1, n-2):
                matrix_big[i, i] = 2*k
                matrix_big[i, i-1] = -1
                matrix_big[i, i+1] = -1
            return matrix_big
        else:
            matrix_big = sp.dok_matrix(((n-1)**l, (n-1)**l))
            identitiy_small = -sp.identity(size)
            matrix_small = Matrix.create_sparse_matrix(l-1, k, n)
            matrix_big[0:size, 0:size] = matrix_small
            matrix_big[0:size, size:2*size] = identitiy_small
            matrix_big[(step-1)*size:step*size, (step-2)*size:(step-1)*size] = identitiy_small
            matrix_big[(step-1)*size:step*size, (step-1)*size:step*size] = matrix_small
            for i in range(1, step-1):
                matrix_big[i*size:(i+1)*size, (i-1)*size:i*size] = identitiy_small
                matrix_big[i*size:(i+1)*size, i*size:(i+1)*size] = matrix_small
                matrix_big[i*size:(i+1)*size, (i+1)*size:(i+2)*size] = identitiy_small
            return matrix_big

    def nonzero_elements(self):
        """
        This method returns the number of nonzero elements in the sparse matrix.
        No inputs are required

        :return: nonzero elements in the sparse Matrix
        :rtype: int
        """
        matrix = self.sparse_matrix()
        nonzero_array = sp.find(matrix)[2]
        return len(nonzero_array)

    def zero_elements(self):
        """
        This method returns the number of zero elements in the sparse matrix.
        No inputs are required

        :return: nonzero elements in the sparse Matrix
        :rtype: int
        """
        nonzero = self.nonzero_elements()
        dim = (self.n-1)**(2*self.d)
        return dim - nonzero

    def relative_nonzero(self):
        """
        This method returns the relative number of nonzero elements to the
        dimension of the sparse matrix.
        No inputs are required

        :return: nonzero elements in the sparse Matrix
        :rtype: float
        """
        nonzero = self.nonzero_elements()
        dim = (self.n-1)**(2*self.d)
        return nonzero/dim

    def relative_zero(self):
        """
        This method returns the relative number of zero elements to the
        dimension of the sparse matrix.
        No inputs are required

        :return: nonzero elements in the sparse Matrix
        :rtype: float
        """
        zero = self.zero_elements()
        dim = (self.n-1)**(2*self.d)
        return zero/dim

    def convert(self):
        """ This method converts the original dok_matrix to Compressed Sparse
        Column format.
        No inputs are required
        
        :return: the converted matrix
        :return type: scipy.sparse
        """
        return self.sparse_matrix().tocsc()

    def inverse(self):
        """ This method calculates the inverse of the sparse matrix.
        No inputs are required
        
        :return: the inverse matrix
        :return type: scipy.sparse
        """
        inv = lg.inv(self.convert())
        if self.n == 2:
            inv = sp.csc_matrix((inv))
        return inv

    def condition(self):
        """ This method calculates the condition number of the sparse matrix.
        No inputs are required.
        
        :return: condition number of the matrix
        :return type: float
        """
        return lg.norm(self.convert(), np.inf)*lg.norm(self.inverse(), np.inf)

    def LU(self):
        """ This method calculates the LU-decomposition of the sparse matrix in
        the same format. It uses an object of SuperLU class which decomposes a
        matrix in the following way:
            Pr*A*Pc = L*U

        where Pr is the column permutation matrix and Pc is the row permutation
        matrix.

        :return: Lower and Upper triangular matrices
         :return type: sparse csc_matrix
        """
        lu = lg.splu(self.convert())
        return [lu.L, lu.U, lu.perm_r, lu.perm_c]

    def LU_solve(self, b, tri=1):
        """
        This method uses an object of Python SuperLU class described in the
        doc-string of the Matrix.LU() method. The algorithm taken to solve the
        system of linear equation is following:
            | Pr*A*Pc = L*U
            | Pr*A*x = L*U*(Pc.T)*x
            | L*U*(Pc.T)*x = Pr*b
            | L*z = Pr*b
            | U*y = z
            | (Pc.T)*x = y

        :param b: the right hand side of equation Ax=b
        :param type: np.array
        :param tri: indicator if spsolve_triangular or spsolve must be used
        :param type: int
        :return: the solved equation for x in np.array
        """
        size = b.shape
        if len(size) == 1:
            b = np.reshape(b, ((self.n-1)**(self.d), 1))
        elif size[0] != (self.n-1)**(self.d) or size[1] != 1:
            raise ValueError("Wrong dimensions for the matrix b")
        L, U, perm_r, perm_c = self.LU()
        Pc = sp.csc_matrix((((self.n)-1)**(self.d), ((self.n)-1)**(self.d)))
        Pr = sp.csc_matrix((((self.n)-1)**(self.d), ((self.n)-1)**(self.d)))
        Pc[np.arange((self.n-1)**(self.d)), perm_c] = 1
        Pr[perm_r, np.arange((self.n-1)**(self.d))] = 1
        b = Pr*b
        if tri == 1:
            z = lg.spsolve_triangular(L, b)
            y = lg.spsolve_triangular(U, z, lower=False)
            x = lg.spsolve(sp.linalg.inv(Pc), y)
        else:
            z = lg.spsolve(L, b)
            y = lg.spsolve(U, z)
            x = lg.spsolve(sp.linalg.inv(Pc), y)
        return np.reshape(x, ((self.n-1)**(self.d), 1))
  
    def cg(self, b, u0 = None, tol = 10**(-8), itmax = None):
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

        if not isinstance(u0, np.ndarray):
            u0 = np.asarray(u0)

        if not isinstance(b, np.ndarray):
            b = np.asarray(b)

        if itmax == None:
            itmax = 1000
        k = 0
        u = u0
        g = self.convert().dot(u0) - b
        d = -g
        while k < itmax:
            alpha = LA.norm(g, 2)**2/(np.dot(d, (self.convert().dot(d)))) 
            if k == 0:
                u_tmp = u + alpha*d
            else:
                u_tmp = u[-1, :] + alpha*d
            u = np.vstack([u, u_tmp])
            g_next = self.convert().dot(u_tmp) - b
            beta = (LA.norm(g_next, 2)/LA.norm(g, 2))**2
            d = -g_next + beta*d
            if LA.norm(g_next, 2) <= tol*LA.norm(g, 2):
                break
            else:
                g = g_next
                k = k + 1
        return u

    def LU_nonzeros(self):
        """
        This method returns the number of nonzero elements in the LU decomposition
        of the sparse coefficient matrix.
        No inputs are required

        :return: nonzero elements in the sparse Matrix
        :rtype: int
        """
        L, U, perm_r, perm_c = self.LU()
        nonzeros_L = sp.find(L)[2]
        nonzeros_U = sp.find(U)[2]
        return len(nonzeros_L) + len(nonzeros_U)

    def LU_relative_nonzeros(self):
        """ This function calculates the relative number of nonzero elements
        of a band matrix.
        No Inputs required.

        :return: the relative number of nonzero elements in the sparse Matrix
        :rtype: float
        """
        m = 2*((self.n - 1)**(2*self.d))
        LU_nonzeros = self.LU_nonzeros()
        return LU_nonzeros/m

    def LU_zeros(self):
        """
        This method returns the number of zero elements in the LU decomposition
        of the sparse coefficient matrix.
        No inputs are required

        :return: nonzero elements in the sparse Matrix
        :rtype: int
        """
        nonzeros = self.LU_nonzeros()
        dim = (self.n-1)**(2*self.d)
        return 2*dim - nonzeros

    def LU_relative_zeros(self):
        """ This function calculates the relative number of zero elements
        of a band matrix.
        No Inputs required.

        :return: the relative number of zero elements in the sparse Matrix
        :rtype: float
        """
        m = 2*((self.n - 1)**(2*self.d))
        LU_zeros = self.LU_zeros()
        return LU_zeros/m

def main():
    """
    In the main program, the functionality of class "Matrix" is tested
    """
    np.set_printoptions(threshold=sys.maxsize)
    n = 3
    d = 2
    matrix = Matrix(d, n)
    a = matrix.convert()
    b = a.toarray()
    print(a, b)
#    print(matrix.convert().toarray())
#    print(matrix.condition())
#    print(matrix.relative_nonzero())
#    print(matrix.relative_zero())
#    print(matrix.LU_relative_nonzeros())
#    print(matrix.LU_relative_zeros())
#    L, U, perm_r, perm_c = matrix.LU()
#    Pr = sp.csc_matrix((((n)-1)**(d), ((n)-1)**(d)))
#    Pr[perm_r, np.arange((n-1)**(d))] = 1
#    L = L.toarray()
#    U = U.toarray()
    b = np.array([1, 1, 1, 1])
#    print(a.dot(b))
    print(matrix.LU_solve(b))
    cg = matrix.cg(b)
    print(cg)

if __name__ == '__main__':
    main()
