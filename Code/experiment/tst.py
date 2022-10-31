# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 18:34:35 2019

@author: ApollouS
"""
import numpy as np

def u(*x):
    """ This is the exact solution of Laplace Equation"""
    u = 1
    for i, j in enumerate(x):
        print(len(x))
        u = u*np.sin(np.pi*x[i])
    return u

def f_exact(*x):
    """ This function calculates the needed right hand side of Laplace Equation
    for the given exact solution u(x)
    """
    u = 1
    for i, j in enumerate(x):
        u = u*np.sin(np.pi*x[i])
    
    return ((np.pi)**2)*len(x)*u

def main():
    print(u(1/2, 1/3, 1/4))
if __name__ == '__main__':
    main()
