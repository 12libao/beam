import itertools
import time

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import scipy
from scipy import sparse, spatial
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, linalg


class Truss:
    def __init__(self, conn, xpts, E=1e2, A=1.0, bcs=[]):
        self.conn = np.array(conn, dtype=int)
        self.xpts = np.array(xpts, dtype=float)
        self.nelems = len(self.conn)
        self.bcs = bcs

        # Set the number of degrees of freedom - 3 dof at each node
        self.ndof = 3 * (np.max(self.conn) + 1)

        self.E = E
        self.rg = 1  # Radius of gyration

        # Set up the i-j indices for the matrix - these are the row
        # and column indices in the stiffness matrix
        self.var = np.zeros((self.conn.shape[0], 6), dtype=int)
        self.var[:, ::3] = 3 * self.conn
        self.var[:, 1::3] = 3 * self.conn + 1
        self.var[:, 2::3] = 3 * self.conn + 2

        i = []
        j = []
        for index in range(self.nelems):
            for ii in self.var[index, :]:
                for jj in self.var[index, :]:
                    i.append(ii)
                    j.append(jj)
        # Convert the lists into numpy arrays
        self.i = np.array(i, dtype=int)
        self.j = np.array(j, dtype=int)

        return

    def get_element_transform(self, elem):
        n1 = self.conn[elem, 0]
        n2 = self.conn[elem, 1]
        dx = self.xpts[n2, 0] - self.xpts[n1, 0]
        dy = self.xpts[n2, 1] - self.xpts[n1, 1]

        L = np.sqrt(dx**2 + dy**2)
        c = dx / L
        s = dy / L

        T = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

        return L, T

    def get_element_stifness_matrix(self, elem, area):
        """Compute the element stiffness matrix"""

        L, T = self.get_element_transform(elem)

        Te = np.zeros((6, 6))
        Te[0:3, 0:3] = T
        Te[3:6, 3:6] = T

        EA = self.E * area
        k1 = EA / L

        EI = self.E * self.rg * area**2
        k2 = EI / L**3

        Ke = np.zeros((6, 6))

        # Set the stiffness matrix for the bar components
        Ke[0, 0] = k1
        Ke[0, 3] = -k1
        Ke[3, 0] = -k1
        Ke[3, 3] = k1

        # Set the elements for the beam components
        Ke[1, 1] = 12 * k2
        Ke[1, 2] = 6 * k2 * L
        Ke[1, 4] = -12 * k2
        Ke[1, 5] = 6 * k2 * L

        Ke[2, 1] = 6 * k2 * L
        Ke[2, 2] = 4 * k2 * L**2
        Ke[2, 4] = -6 * k2 * L
        Ke[2, 5] = 2 * k2 * L**2

        Ke[4, 1] = -12 * k2
        Ke[4, 2] = -6 * k2 * L
        Ke[4, 4] = 12 * k2
        Ke[4, 5] = -6 * k2 * L

        Ke[5, 1] = 6 * k2 * L
        Ke[5, 2] = 2 * k2 * L**2
        Ke[5, 4] = -6 * k2 * L
        Ke[5, 5] = 4 * k2 * L**2

        return Te.T @ Ke @ Te

    def get_element_stress_stifness_matrix(self, elem, u, area):
        """Compute the element stiffness matrix"""

        L, T = self.get_element_transform(elem)

        Te = np.zeros((6, 6))
        Te[0:3, 0:3] = T
        Te[3:6, 3:6] = T

        ue = T @ u

        EA = self.E * area
        k1 = EA / L
        Ne = k1 * (u[3] - u[0])

        EI = self.E * self.rg * area**2
        k2 = EI / L**3

        Ge = np.zeros((6, 6))

        # Set the elements for the beam components
        kg = Ne / L
        Ge[1, 1] = 6 * kg / 5
        Ge[1, 2] = kg * L / 10
        Ge[1, 4] = -6 * kg / 5
        Ge[1, 5] = kg * L / 10

        Ge[2, 1] = kg * L / 10
        Ge[2, 2] = 2 * kg * L**2 / 15
        Ge[2, 4] = -kg * L / 10
        Ge[2, 5] = -kg * L**2 / 30

        Ge[4, 1] = -6 * kg / 5
        Ge[4, 2] = -kg * L / 10
        Ge[4, 4] = 6 * kg / 5
        Ge[4, 5] = -kg * L / 10

        Ge[5, 1] = kg * L / 10
        Ge[5, 2] = -kg * L**2 / 30
        Ge[5, 4] = -kg * L / 10
        Ge[5, 5] = 2 * kg * L**2 / 15

        return Te.T @ Ge @ Te

    def assemble_stiffness_matrix(self, x):
        Ke = np.zeros((self.nelems, 6, 6))

        for elem in range(self.nelems):
            Ke[elem, :, :] = self.get_element_stifness_matrix(elem, x[elem])

        K = sparse.coo_matrix((Ke.flatten(), (self.i, self.j)))
        K = K.tocsr()

        return K

    def assemble_stress_stiffness_matrix(self, x, u):
        Ge = np.zeros((self.nelems, 6, 6))

        for elem in range(self.nelems):
            Ge[elem, :, :] = self.get_element_stress_stifness_matrix(
                elem, x[elem], u[self.var[elem, :]]
            )

        G = sparse.coo_matrix((Ge.flatten(), (self.i, self.j)))
        G = G.tocsr()

        return G
