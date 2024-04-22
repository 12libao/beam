import itertools
import time
from icecream import ic
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
        self.reduced = self._compute_reduced_variables(self.ndof, bcs)
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
            temp[self.reduced] = vec[:]
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

    def _compute_forces(self, ndof, forces):
        """
        Compute the forces vector
        """
        f = np.zeros(ndof)
        for node in forces:
            for index, force in zip(*forces[node]):
                f[3 * node + index] = force
        return f

    def _get_element_transform(self, elem):
        n1 = self.conn[elem, 0]
        n2 = self.conn[elem, 1]
        dx = self.xpts[n2, 0] - self.xpts[n1, 0]
        dy = self.xpts[n2, 1] - self.xpts[n1, 1]

        Le = np.sqrt(dx**2 + dy**2)
        c = dx / Le
        s = dy / Le

        T = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

        return Le, T

    def _get_element_stifness_matrix(self, elem, area):
        """Compute the element stiffness matrix"""

        Le, T = self._get_element_transform(elem)

        Te = np.zeros((6, 6))
        Te[0:3, 0:3] = T
        Te[3:6, 3:6] = T

        EA = self.E * area
        k1 = EA / Le

        EI = self.E * self.rg * area**2
        k2 = EI / Le**3

        Ke = np.zeros((6, 6))

        # Set the stiffness matrix for the bar components
        Ke[0, 0] = k1
        Ke[0, 3] = -k1
        Ke[3, 0] = -k1
        Ke[3, 3] = k1

        # Set the elements for the beam components
        Ke[1, 1] = 12 * k2
        Ke[1, 2] = 6 * k2 * Le
        Ke[1, 4] = -12 * k2
        Ke[1, 5] = 6 * k2 * Le

        Ke[2, 1] = 6 * k2 * Le
        Ke[2, 2] = 4 * k2 * Le**2
        Ke[2, 4] = -6 * k2 * Le
        Ke[2, 5] = 2 * k2 * Le**2

        Ke[4, 1] = -12 * k2
        Ke[4, 2] = -6 * k2 * Le
        Ke[4, 4] = 12 * k2
        Ke[4, 5] = -6 * k2 * Le

        Ke[5, 1] = 6 * k2 * Le
        Ke[5, 2] = 2 * k2 * Le**2
        Ke[5, 4] = -6 * k2 * Le
        Ke[5, 5] = 4 * k2 * Le**2

        return Te.T @ Ke @ Te

    def _get_element_stress_stiffness_matrix(self, elem, u, area):
        """Compute the element stiffness matrix"""

        Le, T = self._get_element_transform(elem)

        Te = np.zeros((6, 6))
        Te[0:3, 0:3] = T
        Te[3:6, 3:6] = T
        ue = Te @ u

        EA = self.E * area
        k1 = EA / Le
        Ne = k1 * (ue[3] - ue[0])

        Ge = np.zeros((6, 6))

        # Set the elements for the beam components
        kg = Ne / Le
        Ge[1, 1] = 6 * kg / 5
        Ge[1, 2] = kg * Le / 10
        Ge[1, 4] = -6 * kg / 5
        Ge[1, 5] = kg * Le / 10

        Ge[2, 1] = kg * Le / 10
        Ge[2, 2] = 2 * kg * Le**2 / 15
        Ge[2, 4] = -kg * Le / 10
        Ge[2, 5] = -kg * Le**2 / 30

        Ge[4, 1] = -6 * kg / 5
        Ge[4, 2] = -kg * Le / 10
        Ge[4, 4] = 6 * kg / 5
        Ge[4, 5] = -kg * Le / 10

        Ge[5, 1] = kg * Le / 10
        Ge[5, 2] = -kg * Le**2 / 30
        Ge[5, 4] = -kg * Le / 10
        Ge[5, 5] = 2 * kg * Le**2 / 15

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

        # ue = np.zeros((self.nelems, 6), dtype=x.dtype)
        u_elem = np.zeros((self.nelems, 6), dtype=x.dtype)

        for elem in range(self.nelems):
            u_elem[elem, 0:3] = u[3 * self.conn[elem, 0] : 3 * self.conn[elem, 0] + 3]
            u_elem[elem, 3:6] = u[3 * self.conn[elem, 1] : 3 * self.conn[elem, 1] + 3]

        for elem in range(self.nelems):
            Ge[elem, :, :] = self._get_element_stress_stiffness_matrix(
                elem, u_elem[elem, :], x[elem]
            )

        G = sparse.coo_matrix((Ge.flatten(), (self.i, self.j)))
        G = G.tocsr()

        return G

    def solve_eigenvalue_problem(self, x, sigma=1.0):

        K0 = self._assemble_stiffness_matrix(x)
        K = self._reduce_matrix(K0)

        # Compute the solution path
        fr = self._reduce_vector(self.f)
        fact = linalg.factorized(K.tocsc())
        u = self._full_vector(fact(fr))

        G0 = self._assemble_stress_stiffness_matrix(x, u)
        G = self._reduce_matrix(G0)

        # mu, Q = scipy.linalg.eigh(G.todense(), K.todense())

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

    def visualize(self, ax, disp=None):
        """
        Visualize the truss structure
        """
        if disp is None:
            disp = np.zeros(self.ndof)

        for elem in range(self.nelems):
            n1 = self.conn[elem, 0]
            n2 = self.conn[elem, 1]

            x1 = self.xpts[n1, 0] + disp[3 * n1]
            y1 = self.xpts[n1, 1] + disp[3 * n1 + 1]
            x2 = self.xpts[n2, 0] + disp[3 * n2]
            y2 = self.xpts[n2, 1] + disp[3 * n2 + 1]

            ax.plot([x1, x2], [y1, y2], "b-o")

        return


if __name__ == "__main__":
    # Set the random seed
    np.random.seed(0)

    # set the beam parameters use dictioanry
    settings = {
        "L": 1.0,
        "nelems": 20,
        "N": 6,
        "E": 1.0,
        "rg": 1.0,
        "rho": 1.0,
        "ks_rho": 10.0,
    }

    # Set the length of the beam
    L = settings["L"]
    nelems = settings["nelems"]
    N = settings["N"]

    # Set the nodal coordinates
    xpts = np.zeros((nelems + 1, 2))
    xpts[:, 0] = np.linspace(0.0, L, nelems + 1)

    # Set the connectivity
    conn = np.array([[i, i + 1] for i in range(nelems)])

    # Set the boundary conditions, Fixed at the left end and pinned at the right end
    bcs = {0: [0, 1], nelems: [1]}

    # Set the forces, tip force at the right end
    forces = {nelems: ([0], [-1])}

    # Create the truss object
    truss = Truss(
        conn, xpts, E=settings["E"], rg=settings["rg"], bcs=bcs, forces=forces, N=N
    )

    # visualize the truss with displacement
    x = np.ones(nelems)
    BLF, Q = truss.solve_eigenvalue_problem(x)

    BLF_al = np.pi**2 * settings["E"] * settings["rg"] * x[0] ** 2 / settings["L"] ** 2
    ic(BLF[0] - BLF_al)

    fig, ax = plt.subplots(N, 1, figsize=(4, N), sharex=True, sharey=True)
    for i in range(N):
        truss.visualize(ax[i], disp=Q[:, i])

    plt.show()
