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
    def __init__(
        self,
        conn,
        xpts,
        E=1.0,
        rg=1.0,
        hdof=0,
        N=10,
        bcs=[],
        forces={},
    ):
        self.conn = np.array(conn, dtype=int)
        self.xpts = np.array(xpts, dtype=float)
        self.nelems = len(self.conn)
        self.bcs = bcs

        # Degree of freedom to use as a relative modal displacement constraint
        self.hdof = hdof

        # Number of eigenvectors to compute
        self.N = N

        # Set the number of degrees of freedom - 3 dof at each node
        self.ndof = 3 * (np.max(self.conn) + 1)

        self.E = E  # Elastic modulus
        self.rg = rg  # Radius of gyration

        # Set the reduced set of forces
        self.reduced = self._compute_reduced_variables(self.nvndofars, bcs)
        self.f = self._compute_forces(self.ndof, forces)

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

    def _compute_reduced_variables(self, ndof, bcs):
        """
        Compute the reduced set of variables
        """
        reduced = list(range(ndof))

        # For each node that is in the boundary condition dictionary
        for node in bcs:
            dof_list = bcs[node]

            # For each index in the boundary conditions (corresponding to
            # either a constraint on u and/or constraint on v
            for index in dof_list:
                var = 3 * node + index
                reduced.remove(var)

        return reduced

    def _reduce_vector(self, forces):
        """
        Eliminate essential boundary conditions from the vector
        """
        return forces[self.reduced]

    def _reduce_matrix(self, matrix):
        """
        Eliminate essential boundary conditions from the matrix
        """
        temp = matrix[self.reduced, :]
        return temp[:, self.reduced]

    def _full_vector(self, vec):
        """
        Transform from a reduced vector without dirichlet BCs to the full vector
        """
        if vec.ndim == 1:
            temp = np.zeros(self.ndof, dtype=vec.dtype)
            temp[self.reduced, :] = vec[:]
        elif vec.ndim == 2:
            temp = np.zeros((self.ndof, vec.shape[1]), dtype=vec.dtype)
            temp[self.reduced, :] = vec[:]
        return temp

    def _full_matrix(self, mat):
        """
        Transform from a reduced matrix without dirichlet BCs to the full matrix
        """
        temp = np.zeros((self.ndof, self.ndof), dtype=mat.dtype)
        for i in range(len(self.reduced)):
            for j in range(len(self.reduced)):
                temp[self.reduced[i], self.reduced[j]] = mat[i, j]
        return temp

    def _get_element_transform(self, elem):
        n1 = self.conn[elem, 0]
        n2 = self.conn[elem, 1]
        dx = self.xpts[n2, 0] - self.xpts[n1, 0]
        dy = self.xpts[n2, 1] - self.xpts[n1, 1]

        L = np.sqrt(dx**2 + dy**2)
        c = dx / L
        s = dy / L

        T = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

        return L, T

    def _get_element_stifness_matrix(self, elem, area):
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

    def _get_element_stress_stifness_matrix(self, elem, u, area):
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

    def _assemble_stiffness_matrix(self, x):
        Ke = np.zeros((self.nelems, 6, 6))

        for elem in range(self.nelems):
            Ke[elem, :, :] = self._get_element_stifness_matrix(elem, x[elem])

        K = sparse.coo_matrix((Ke.flatten(), (self.i, self.j)))
        K = K.tocsr()

        return K

    def _assemble_stress_stiffness_matrix(self, x, u):
        Ge = np.zeros((self.nelems, 6, 6))

        for elem in range(self.nelems):
            Ge[elem, :, :] = self._get_element_stress_stifness_matrix(
                elem, x[elem], u[self.var[elem, :]]
            )

        G = sparse.coo_matrix((Ge.flatten(), (self.i, self.j)))
        G = G.tocsr()

        return G

    def solve_eigenvalue_problem(self, x, sigma=1.0):

        K0 = self._assemble_stiffness_matrix(x)
        K = self._reduce_matrix(K0)

        # Compute the solution path
        fr = self._reduce_vector(self.f)
        fact = linalg.factorized(K0)
        u = self._full_vector(fact(fr))

        G0 = self._assemble_stress_stiffness_matrix(x)
        G = self._reduce_matrix(G0)

        mu, Q = sparse.linalg.eigsh(
            G,
            M=K,
            k=self.N,
            sigma=sigma,
            which="SM",
            maxiter=1000,
            tol=1e-8,
        )

        self.mu = mu
        self.BLF = -1.0 / mu
        self.Q = self._full_vector(Q)

        return self.BLF, self.Q

    def buckling_aggregate(self, x):

        BLF, Q = self.solve_eigenvalue_problem(x)

        eta = np.exp(-self.ks_rho * (BLF - np.min(BLF)))
        eta = eta / np.sum(eta)

        h = 0.0
        for i in range(self.N):
            h += eta[i] * Q[self.hdof] ** 2

        return h
