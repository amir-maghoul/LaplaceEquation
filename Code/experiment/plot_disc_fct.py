""" This module provides the functionality to plot a discrete function on the unit square.

Author: Franz Bethke
"""

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def vec_idx_to_msh_idx(vec_idx, disc):
    """ Computes msh indices for given vector indices.
    Input:
        vec_idx (np.array):
        disc (int): specifies a mesh with (`disc`-1)**2 interior points.
    Return:
        (tuple(np.array, np.array)): (column of the point in the mesh,
                                      row of the point in the mesh)
    """
    col_idx = vec_idx%(disc+1)
    row_idx = vec_idx//(disc+1)
    return col_idx, row_idx


def on_boundary(disc):
    """ Checks for every point in the mesh if it is on the boundary.
    Input:
        disc (int): specifies a mesh with (`disc`-1)**2 interior points.
    Reurn:
        (np.array of bool): `True` entries for boundary points.
    """
    # construct list of all vector indices and compute correpsonding mesh indicies
    vec_idx = np.arange((disc+1)**2)
    col_idx, row_idx = vec_idx_to_msh_idx(vec_idx, disc)
    # check boundary conditions
    return (col_idx == 0) | (col_idx == disc) | (row_idx == 0) | (row_idx == disc)


def plot_disc_fct(fct, disc, axis, title=""):
    """ Plots a discrete function with zero boundary data.

    Input:
        fct (np.array): function values on the interior points of the mesh.
        disc (int): specifies a mesh with (`disc`-1)**2 interior points.
        axis: axis Object of the plot
        title (str): title of the figure
    """

    # find interior and boundary points
    boundary = on_boundary(disc)
    interior = ~boundary

    # pad function values with zeros (boundary data)
    tmp = np.zeros(((disc+1)**2))
    tmp[interior] = fct
    fct = tmp.reshape(disc+1, disc+1)

    # build the mesh
    x1_msh = np.linspace(0, 1, disc+1)
    x2_msh = np.linspace(0, 1, disc+1)
    x1_msh, x2_msh = np.meshgrid(x1_msh, x2_msh)


    surf = axis.plot_surface(x1_msh, x2_msh, fct, cmap=mpl.cm.coolwarm)
    axis.view_init(20, -105)

    # set plot settings
    plt.xlabel('x1')
    plt.ylabel('x2')
#    fig.colorbar(surf, shrink=0.5)
    plt.title(title)
    return surf


def test():
    """ A simple test case for the functions in this module.
    """
    fig = plt.figure()
    axis = Axes3D(fig)
    
    disc = 5

    # this could be the solution x of A*x = b
    fct = [1, 2, 3, 4,
           2, 3, 4, 5,
           5, 6, 7, 8,
           9, 10, 11, 12]

    plot_disc_fct(fct, disc, axis, "Testplot")

    plt.show()


if __name__ == "__main__":
    test()
