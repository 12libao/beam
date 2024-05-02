import itertools
import time

from icecream import ic
import matplotlib
import matplotlib as mpl
from matplotlib import patheffects
import matplotlib.gridspec as gridspec
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.patheffects as patheffects
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import scipy
from scipy import sparse, spatial
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, linalg

matplotlib.use("ps")
matplotlib.rcParams.update({"text.usetex": True})
# matplotlib.rc("text.latex", preamble=r"\usepackage{xcolor}")
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    }
)

plt.rcParams["text.usetex"] = True
plt.rcParams[
    "text.latex.preamble"
] = r"""
\usepackage[english]{babel}
\usepackage[utf8x]{inputenc}
\usepackage{mathtools}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{xcolor}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
% more packages here
"""

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{xcolor} "

niceColors = dict()
niceColors["Yellow"] = "#f8a30d"
niceColors["Yellow-light"] = "#fbcf83"
niceColors["Blue"] = "#1E90FF"
niceColors["Blue-light"] = "#99ccff"
niceColors["Red"] = "#E21A1A"
niceColors["Red-light"] = "#f4a4a4"
niceColors["Green"] = "#00a650"
niceColors["Green-light"] = "#99ffca"
niceColors["Maroon"] = "#800000"
niceColors["Maroon-light"] = "#f6b5a2"
niceColors["Orange"] = "#E64616"
niceColors["Orange-light"] = "#f6b5a2"
niceColors["Purple"] = "#800080"
niceColors["Purple-light"] = "#ff99ff"
niceColors["Cyan"] = "#00A6D6"
niceColors["Cyan-light"] = "#99e9ff"
niceColors["Grey"] = "#5a5758"
niceColors["Black"] = "#000000"
colors = list(niceColors.values())

title_fontsize = 8
label_fontsize = 8
legend_fontsize = 7
text_fontsize = 5.5


class Truss:
    def __init__(
        self,
        conn,
        xpts,
        E=1.0,
        rg=1.0,
        N=10,
        bcs=[],
        forces={},
        hdof=None,
    ):
        self.conn = np.array(conn, dtype=int)
        self.xpts = np.array(xpts, dtype=float)
        self.nelems = len(self.conn)
        self.bcs = bcs
        self.nn = self.nelems // 6  # Number of elements on each side

        # Number of eigenvectors to compute
        self.N = N

        # Set the number of degrees of freedom - 3 dof at each node
        self.ndof = 3 * (np.max(self.conn) + 1)

        # Degree of freedom to use as a relative modal displacement constraint
        if hdof is None:
            self.hdof = np.zeros((self.ndof, self.ndof))
            node_index = np.floor(self.nelems / 2).astype(int)
            ic(node_index)
            self.hdof[3 * (node_index - 1), 3 * (node_index - 1)] = 1.0
            # self.hdof[3 * (node_index - 1) + 1, 3 * (node_index - 1) + 1] = 1.0
            # self.hdof[3 * (node_index - 1) + 2, 3 * (node_index - 1) + 2] = 1.0
        else:
            self.hdof = hdof
            ic(self.hdof)

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
            forces: np.array([nnodes, 3])
        """
        f = np.zeros(ndof, dtype=float)
        for i in range(forces.shape[0]):
            f[3 * i : 3 * i + 3] = forces[i, :]

        return f

    def _get_element_transform(self, elem):
        n1 = self.conn[elem, 0]
        n2 = self.conn[elem, 1]
        dx = self.xpts[n2, 0] - self.xpts[n1, 0]
        dy = self.xpts[n2, 1] - self.xpts[n1, 1]

        Le = np.sqrt(dx**2 + dy**2)
        c = dx / Le
        s = dy / Le

        T = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])

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
        Ge[1, 1] = 6 / 5
        Ge[1, 2] = Le / 10
        Ge[1, 4] = -6 / 5
        Ge[1, 5] = Le / 10

        Ge[2, 1] = Le / 10
        Ge[2, 2] = 2 * Le**2 / 15
        Ge[2, 4] = -Le / 10
        Ge[2, 5] = -(Le**2) / 30

        Ge[4, 1] = -6 / 5
        Ge[4, 2] = -Le / 10
        Ge[4, 4] = 6 / 5
        Ge[4, 5] = -Le / 10

        Ge[5, 1] = Le / 10
        Ge[5, 2] = -(Le**2) / 30
        Ge[5, 4] = -Le / 10
        Ge[5, 5] = 2 * Le**2 / 15

        kg = Ne / Le
        Ge = kg * Ge

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

    def solve_deformation(self, x):
        K0 = self._assemble_stiffness_matrix(x)
        K = self._reduce_matrix(K0)

        # Compute the solution path
        fr = self._reduce_vector(self.f)
        fact = linalg.factorized(K.tocsc())
        u = self._full_vector(fact(fr))

        return u

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

    def buckling_aggregate(self, x, ks_rho_h=1.0):

        BLF, Q = self.solve_eigenvalue_problem(x)

        eta = np.exp(-ks_rho_h * (BLF - np.min(BLF)))
        eta = eta / np.sum(eta)

        h = 0.0
        for i in range(self.N):
            h += eta[i] * Q[:, i].T @ self.hdof @ Q[:, i]

        return h

    def approx_eigenvalue(self, x, sigma=1.0, ks_rho_blf=30.0):
        """
        Approximate minimum eigenvalue using the ks aggregation
        """
        BLF, _ = self.solve_eigenvalue_problem(x, sigma=sigma)

        # min_blf = np.min(BLF)
        # eta = np.exp(-ks_rho_blf * (BLF - min_blf))

        # return min_blf - np.log(np.sum(eta)) / ks_rho_blf

        mu = 1.0 / BLF
        c = np.max(mu)
        eta = np.exp(ks_rho_blf * (mu - c))
        return c + np.log(np.sum(eta)) / ks_rho_blf


class Optimization(Truss):
    def __init__(self, conn, xpts, E=1.0, rg=1.0, N=10, bcs=[], forces={}, **kwargs):

        super().__init__(conn, xpts, E, rg, N, bcs, forces)

        self.ks_rho_h = kwargs.get("ks_rho_h", 10.0)
        self.ks_rho_blf = kwargs.get("ks_rho_blf", 10.0)
        self.mass_upper = 1.1
        self.h_upper = 0.23

    def objective(self, x):
        area = self.construst(x)
        return self.approx_eigenvalue(area, self.ks_rho_blf)

    def mass_constr(self, x):
        return self.mass_upper - x[0] - x[1]

    def mass_constr_grad(self, x):
        return np.array([-1, -1])

    def h_constr(self, x):
        area = self.construst(x)
        h = self.buckling_aggregate(area, self.ks_rho_h)
        return self.h_upper - h

    def construst(self, x):
        nn = self.nelems // 6
        area = np.zeros(self.nelems)
        area[: 4 * nn] = x[0]
        area[4 * nn :] = x[1]

        # area[:nn] = x[0]
        # area[2 * nn : 3 * nn] = x[0]
        # area[nn : 2 * nn] = 1 - x[0] - x[1]
        # area[3 * nn : 4 * nn] = 1 - x[0] - x[1]

        return area

    def optimize(self, h_upper=0.23):
        x0 = [0.1, 0.1]
        self.x_hist = []
        self.x_hist.append(x0)
        self.h_upper = h_upper

        constr = (
            {"type": "ineq", "fun": self.mass_constr, "jac": self.mass_constr_grad},
        )
        if self.h_upper < 1e3:
            constr = (
                {"type": "ineq", "fun": self.mass_constr, "jac": self.mass_constr_grad},
                {"type": "ineq", "fun": self.h_constr},
            )

        res = minimize(
            self.objective,
            x0,
            method="SLSQP",  # "trust-constr", "COBYLA", "L-BFGS-B", "SLSQP", "trust-ncg"
            bounds=[(0.001, 1.2), (0.001, 1.2)],
            constraints=constr,
            options={"disp": True, "maxiter": 200, "ftol": 1e-8},
            callback=lambda x: print(
                "obj: ",
                self.objective(x),
                " h_constr: ",
                self.h_constr(x),
                " mass_constr: ",
                self.mass_constr(x),
            )
            or self.x_hist.append(x),
        )

        x_hist = np.array(self.x_hist)

        return res, x_hist

    def plot_contour(self, gs, fig, x0, x1, x2, h1, h2, n=20):
        x = np.linspace(0.01, 1.2, n)
        y = np.linspace(0.01, 1.5, n)
        X, Y = np.meshgrid(x, y)

        # cl= ["r", "sandybrown", colors[3]]
        # cl = ["k", "royalblue", "sandybrown"]
        cl = ["k", "k", "r"]

        # Z = np.zeros(X.shape)
        # mv = np.zeros(X.shape)
        # hv1 = np.zeros(X.shape)
        # hv2 = np.zeros(X.shape)

        # for i in range(X.shape[0]):
        #     for j in range(X.shape[1]):
        #         Z[i, j] = self.objective([X[i, j], Y[i, j]])
        #         mv[i, j] = self.mass_constr([X[i, j], Y[i, j]])

        #         self.h_upper = h1
        #         hv1[i, j] = self.h_constr([X[i, j], Y[i, j]])

        #         self.h_upper = h2
        #         hv2[i, j] = self.h_constr([X[i, j], Y[i, j]])

        # # sava the Z, mv, hv1, hv2
        # np.save("Z.npy", Z)
        # np.save("mv.npy", mv)
        # np.save("hv1.npy", hv1)
        # np.save("hv2.npy", hv2)

        # read the Z, mv, hv1, hv2
        Z = np.load("Z.npy")
        mv = np.load("mv.npy")
        hv1 = np.load("hv1.npy")
        hv2 = np.load("hv2.npy")

        # normalize the Z inbetween 0 and 1
        Z1 = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))

        # fig, ax = plt.subplots(1, 2, figsize=(7.48, 2.8), sharex=True, sharey=True)
        # devide the figure into two parts

        # N1GS2 = gs.subgridspec(1, 2, width_ratios=[1, 1.15])
        N1GS2 = gs.subgridspec(1, 2, width_ratios=[1, 1])
        # add constrainted design space
        ax = [fig.add_subplot(N1GS2[0, 0]), fig.add_subplot(N1GS2[0, 1])]
        
        # share the y axis for the two subplots
        ax[1].sharey(ax[0])

        # fig.subplots_adjust(wspace=0.1)
        base_colors = "coolwarm"

        for i in range(2):
            cs = ax[i].contourf(
                X, Y, np.log(Z1), levels=50, alpha=0.2, cmap=base_colors
            )
            ax[i].contour(
                X,
                Y,
                np.log(Z1),
                levels=50,
                alpha=1.0,
                cmap=base_colors,
                linewidths=0.4,
            )

        # add local minimum
        l0, l1 = [], []
        for k in range(2):
            n_loc_min = [2, 3][k]
            for i in range(n_loc_min):
                x = [x0, x1, x2][i]
                c_h = ["k", "r", "b"][i]
                a = [[1.0, 0.5, 0.25], [0.25, 1.0, 1.0]][k][i]
                p0 = ax[k].plot(
                    x[0],
                    x[1],
                    "*",
                    ms=10,
                    markerfacecolor=c_h,
                    markeredgewidth=0.0,
                    zorder=100,
                    alpha=a,
                )
                ax[k].contour(
                    X,
                    Y,
                    np.log(Z),
                    [np.log(self.objective(x))],
                    colors=c_h,
                    linewidths=1.0,
                    linestyles="-",
                    alpha=a,
                )
                p0 = plt.plot(
                    [], [], "*", color=c_h, ms=5, markeredgewidth=0.0, alpha=a
                )
                p1 = plt.plot([], [], "-", color=c_h, linewidth=1.0, alpha=a)
                list = [l0, l1][k]
                list.append(p0[0])
                list.append(p1[0])

        # add mass constraint line
        for i in range(2):
            cg1 = ax[i].contour(X, Y, -mv, [0.0], colors="k", alpha=1.0, linewidths=0.5)
            plt.setp(
                cg1.collections,
                path_effects=[patheffects.withTickedStroke(length=1.0, spacing=4)],
            )
            p2 = plt.plot([], [], "-", color="k", linewidth=0.5)
            [l0, l1][i].append(p2[0])

        # add h-constraint line for ax[1]
        for i, h in enumerate([h1, h2]):
            hv = [hv1, hv2][i]
            c_h = ["r", "b"][i]
            cg2 = ax[1].contour(X, Y, -hv, [0.0], colors=c_h, linewidths=0.5)
            plt.setp(
                cg2.collections,
                path_effects=[patheffects.withTickedStroke(length=1.0, spacing=4)],
                label=r"$h$-constraint ($\bar{h}$ = " + str(h) + ")",
            )
            p3 = plt.plot([], [], "-", color=c_h, linewidth=0.5)
            l1.append(p3[0])

        ax[0].legend(
            [(l0[0], l0[2]), (l0[1], l0[3]), l0[4]],
            [
                r"Local minimums $x^{*}_{i}, x^{*}_{ii}$",
                r"Contour for $x^{*}_{i}, x^{*}_{ii}$",
                "Mass constraint",
            ],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc="upper right",
            fontsize=legend_fontsize,
            frameon=False,
        )
        ax[1].legend(
            [(l1[0], l1[2], l1[4]), (l1[1], l1[3], l1[5]), l1[6], l1[7], l1[8]],
            [
                r"Local minimums $x^{*}_{i}, x^{*}_{ii}, x^{*}_{iii}$",
                r"Contour for $x^{*}_{i}, x^{*}_{ii}, x^{*}_{iii}$",
                "Mass constraint",
                r"$h$-constraint ($\bar{h}$ = " + str(h1) + ")",
                r"$h$-constraint ($\bar{h}$ = " + str(h2) + ")",
            ],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            handlelength=3,
            loc="upper right",
            fontsize=legend_fontsize,
            frameon=False,
        )

        # add quiver
        ax1 = [0.1, 0.05]
        ax2 = [0.53, 0.48]
        dx = (ax2[0] - ax1[0]) * 18
        dy = (ax2[1] - ax1[1]) * 18
        Q = ax[0].quiver(
            ax1[0],
            ax1[1],
            dx,
            dy,
            alpha=1.0,
            scale=10,
            scale_units="inches",
            width=0.004,
            zorder=100,
        )
        ax[0].text(
            ax1[0] - 0.025,
            ax1[1] + 0.025,
            r"Direction of larger $BLF_{1}$",
            fontsize=7,
            color="k",
            ha="left",
            va="bottom",
            rotation=45,
            zorder=100,
        )

        ax[0].set_xlabel(r"$x_1$", fontsize=label_fontsize)
        ax[1].set_xlabel(r"$x_1$", fontsize=label_fontsize)
        ax[0].set_ylabel(r"$x_2$", fontsize=label_fontsize)
        # set the colorbar is not included in ax[1]
        cbar = fig.colorbar(
            cs, ax=ax[1], orientation="vertical", pad=0.02
        )  
        cbar.set_label(r"Scaled $J^{KS}[1/BLF]$")
        cbar.ax.yaxis.labelpad = 1.0
        cbar.ax.tick_params(labelsize=7)
        cbar.ax.yaxis.set_label_position("right")

        ax[0].set_title(
            r"Design Space no Relative Displacement Constraint", fontsize=title_fontsize
        )
        ax[1].set_title(
            r"Design Space with Relative Displacement Constraint",
            fontsize=title_fontsize,
        )

        # make sure x and y axis are equal
        # ax[0].set_aspect("equal", "box")
        # ax[1].set_aspect("equal", "box")

        return

    def generate_min_path(self, n):
        x_min1 = np.zeros((n, 2))
        x_min2 = np.zeros((n, 2))

        mass = np.linspace(0.8, 2.0, n)

        for i in range(n):
            self.mass_upper = mass[i]

            res = self.optimize(h_constr=False)
            x_min1[i, :] = res.x

            res = self.optimize(h_constr=True)
            x_min2[i, :] = res.x

        return x_min1, x_min2


class Domain:
    def __init__(self, nelems, L, f):
        self.nelems = nelems
        self.L = L
        self.f = f

        self.ndof = 0
        self.xpts = []
        self.conn = []
        self.bcs = {}
        self.forces = {}

        return

    def domain_bar(self):
        """
        Create the simple domain for the truss problem
        """
        # Set the nodal coordinates
        self.xpts = np.zeros((self.nelems + 1, 2))
        self.xpts[:, 0] = np.linspace(0.0, self.L, self.nelems + 1)

        # Set the connectivity
        self.conn = np.array([[i, i + 1] for i in range(self.nelems)])

        # Set the boundary conditions, Fixed at the left end and pinned at the right end
        self.bcs = {0: [0, 1], self.nelems: [1]}

        # Set the forces, tip force at the right end
        self.forces = {self.nelems: ([0], [-1])}

        # BLF_al = np.pi**2 * settings["E"] * settings["rg"] * x[0] ** 2 / L**2
        # ic(BLF[0] - BLF_al)

        # fig, ax = plt.subplots(N, 1, figsize=(4, 4), sharex=True, sharey=True)
        # for i in range(N):
        #     domain.visualize(ax[i], disp=Q[:, i])

        # plt.show()

        return self.conn, self.xpts, self.bcs, self.forces

    def domain_square(self):
        """
        Create a simple domain for the truss problem
        """
        # Set the number of elements to be even
        if self.nelems % 12 != 0:
            self.nelems = self.nelems - self.nelems % 12 + 12

        nn = self.nelems // 6  # Number of elements on each side
        Le = self.L / nn

        # set the nodal coordinates for the edge of the square along counter clockwise
        for i in range(nn + 1):
            self.xpts.append([i * Le, 0])
        for i in range(1, nn + 1):
            self.xpts.append([self.L, i * Le])
        for i in range(nn - 1, -1, -1):
            self.xpts.append([i * Le, self.L])
        for i in range(nn - 1, 0, -1):
            self.xpts.append([0, i * Le])

        # Set the nodal coordinates
        for i, (x, y) in enumerate(itertools.product(range(nn + 1), repeat=2)):
            if (x == y or x + y == nn) and (x != 0 and x != nn):
                self.xpts.append([y * Le, x * Le])

        self.xpts = np.array(self.xpts)
        self.nnodes = self.xpts.shape[0]
        self.ndof = 3 * self.nnodes

        # connect the nodes
        self.conn = np.array([[i, i + 1] for i in range(4 * nn)])
        self.conn[-1, 1] = 0

        # connect the diagonals
        center = 5 * nn - 2
        for i in range(4 * nn, 6 * nn - 4, 2):
            self.conn = np.append(self.conn, [[i, i + 2]], axis=0)

        for i in range(4 * nn + 1, 6 * nn - 5, 2):
            if i == center - 1:
                self.conn = np.append(self.conn, [[i, i + 1]], axis=0)
                self.conn = np.append(self.conn, [[i + 1, i + 2]], axis=0)
            else:
                self.conn = np.append(self.conn, [[i, i + 2]], axis=0)

        # connect the four corners to daiagonals
        self.conn = np.append(self.conn, [[0, 4 * nn]], axis=0)
        self.conn = np.append(self.conn, [[nn, np.min((4 * nn + 1, center))]], axis=0)
        self.conn = np.append(self.conn, [[2 * nn, 6 * nn - 4]], axis=0)
        self.conn = np.append(
            self.conn, [[3 * nn, np.max((6 * nn - 5, center))]], axis=0
        )

        # Set the boundary conditions
        self.bcs = {
            nn // 2: [0],
            1 * nn + nn // 2: [1],
            2 * nn + nn // 2: [0],
            3 * nn + nn // 2: [1],
        }

        # Set the forces
        self.forces = np.zeros((self.nnodes, 3))
        self.forces[nn // 2, 1] = self.f  # bottom side
        self.forces[1 * nn + nn // 2, 0] = -self.f  # right side
        self.forces[2 * nn + nn // 2, 1] = -self.f  # top side
        self.forces[3 * nn + nn // 2, 0] = self.f  # left side

        # self.forces[0, :2] = self.f / np.sqrt(2)
        # self.forces[nn, 0] = -self.f / np.sqrt(2)
        # self.forces[nn, 1] = self.f / np.sqrt(2)
        # self.forces[2 * nn, :2] = -self.f / np.sqrt(2)
        # self.forces[3 * nn, 0] = self.f / np.sqrt(2)
        # self.forces[3 * nn, 1] = -self.f / np.sqrt(2)

        return self.conn, self.xpts, self.bcs, self.forces

    def domain_diamond(self):
        self.conn, self.xpts, self.bcs, self.forces = self.domain_square()

        # rotate the coordinates 45 degrees and rescacle
        for i in range(self.nnodes):
            x = self.xpts[i, 0]
            y = self.xpts[i, 1]
            self.xpts[i, 0] = x + y
            self.xpts[i, 1] = y - x

        # reset the origin to the center of the square
        self.xpts = self.xpts - np.mean(self.xpts, axis=0)

        # reset the bcs and forces
        nn = self.nelems // 6
        corner1 = 0
        corner2 = nn
        corner3 = 2 * nn
        corner4 = 3 * nn

        self.bcs = {
            corner1: [1],
            corner2: [0],
            corner3: [1],
            corner4: [0],
        }

        self.forces = np.zeros((self.nnodes, 3))
        self.forces[corner1, 0] = self.f
        self.forces[corner2, 1] = self.f
        self.forces[corner3, 0] = -self.f
        self.forces[corner4, 1] = -self.f

        return self.conn, self.xpts, self.bcs, self.forces

    def domain_circle(self):
        """
        Create a simple domain for the truss problem
        """

        self.conn, self.xpts, self.bcs, self.forces = self.domain_square()

        # 4 edges: reset xpts as a circle and rotate -90 degrees
        nn = self.nelems // 6  # Number of elements on each side
        for i in range(4 * nn):
            self.xpts[i, 0] = np.sin(2 * np.pi * i / (4 * nn))
            self.xpts[i, 1] = -np.cos(2 * np.pi * i / (4 * nn))

        # 2 diagonals transformate
        for i in range(4 * nn, self.nnodes):
            x = self.xpts[i, 0]
            y = self.xpts[i, 1]
            self.xpts[i, 0] = x - y
            self.xpts[i, 1] = x + y - 1.0

        self.bcs = {
            0: [0],
            1 * nn: [1],
            2 * nn: [0],
            3 * nn: [1],
            # 5*nn-2: [0, 1],
        }

        self.forces = np.zeros((self.nnodes, 3))
        self.forces[nn // 2, 0] = -self.f
        self.forces[nn // 2, 1] = self.f
        self.forces[1 * nn + nn // 2, :2] = -self.f
        self.forces[2 * nn + nn // 2, 0] = self.f
        self.forces[2 * nn + nn // 2, 1] = -self.f
        self.forces[3 * nn + nn // 2, :2] = self.f

        # self.forces = np.zeros((self.nnodes, 3))
        # self.forces[0, 1] = self.f  # left corner along x-axis
        # self.forces[nn, 0] = -self.f
        # self.forces[2 * nn, 1] = -self.f
        # self.forces[3 * nn, 0] = self.f

        return self.conn, self.xpts, self.bcs, self.forces

    def domain_square_mesh(self):
        """
        Create a simple domain for the truss problem
        """

        nn = self.nelems  # Number of elements on each side
        Le = self.L / nn

        self.nelems = nn * (nn + 1) * 2
        self.nnodes = (nn + 1) * (nn + 1)
        self.ndof = 3 * self.nnodes

        # create the mesh
        x = np.linspace(0, self.L, nn + 1)
        y = np.linspace(0, self.L, nn + 1)
        X, Y = np.meshgrid(x, y)

        self.xpts = np.array([X.flatten(), Y.flatten()]).T

        # Set the connectivity
        self.conn = []
        for i in range(nn):
            for j in range(nn):
                node = i * (nn + 1) + j
                self.conn.append([node, node + 1])
                self.conn.append([node, node + nn + 1])

            # connect the last row
            node = nn * (nn + 1) + i
            self.conn.append([node, node + 1])

            # connect the last column
            node = i * (nn + 1) + nn
            self.conn.append([node, node + nn + 1])

        self.conn = np.array(self.conn)

        # Set the boundary conditions
        self.bcs = {
            0: [0, 1],
        }

        # Set the forces
        self.forces = np.zeros((self.nnodes, 3))

        return self.conn, self.xpts, self.bcs, self.forces

    def domain_diamond_mesh(self):
        self.conn, self.xpts, self.bcs, self.forces = self.domain_square_mesh()

        # rotate the coordinates 45 degrees and rescacle
        for i in range(self.nnodes):
            x = self.xpts[i, 0]
            y = self.xpts[i, 1]
            self.xpts[i, 0] = x + y
            self.xpts[i, 1] = y - x

        # reset the origin to the center of the square
        self.xpts = self.xpts - np.mean(self.xpts, axis=0)

        # reset the bcs and forces
        nn = int(np.sqrt(self.nnodes) - 1)

        corner1 = 0
        corner2 = nn
        corner3 = self.nnodes - 1
        corner4 = self.nnodes - nn - 1

        self.bcs = {
            corner1: [1],
            corner2: [0],
            corner3: [1],
            corner4: [0],
        }

        self.forces = np.zeros((self.nnodes, 3))
        self.forces[corner1, 0] = self.f
        self.forces[corner2, 1] = self.f
        self.forces[corner3, 0] = -self.f
        self.forces[corner4, 1] = -self.f

        return self.conn, self.xpts, self.bcs, self.forces

    def visualize(self, ax, disp=None, x=None, c=None, quiver=False, domain=False):

        if c is None:
            c = "k"

        if disp is None:
            disp = np.zeros(self.ndof)

        for elem in range(self.nelems):
            n1 = self.conn[elem, 0]
            n2 = self.conn[elem, 1]

            x10 = self.xpts[n1, 0]
            y10 = self.xpts[n1, 1]
            x20 = self.xpts[n2, 0]
            y20 = self.xpts[n2, 1]

            x1 = x10 + disp[3 * n1]
            y1 = y10 + disp[3 * n1 + 1]
            x2 = x20 + disp[3 * n2]
            y2 = y20 + disp[3 * n2 + 1]

            if x is None and not domain:
                ax.plot([x10, x20], [y10, y20], "k--", lw=0.5, alpha=0.2)
                ax.plot([x1, x2], [y1, y2], "-", lw=0.5, color=c)
            elif domain and x is None:
                cw = plt.colormaps["coolwarm"](np.linspace(0, 1, 10))
                if elem < 2 * self.nelems // 3:
                    self.P_x1 = ax.plot(
                        [x1, x2], [y1, y2], "-o", lw=0.5, ms=2, color=cw[0]
                    )
                else:
                    self.P_x2 = ax.plot(
                        [x1, x2], [y1, y2], "-o", lw=0.5, ms=2, color=cw[9]
                    )
            else:
                ax.plot([x10, x20], [y10, y20], "k--", lw=4 * x[elem], alpha=0.2)
                ax.plot([x1, x2], [y1, y2], "-", lw=4 * x[elem], color=c)

            ax.axis("equal")
            ax.axis("off")

        if x is None:
            if domain:
                ms = 5
            else:
                ms = 2
            # add the boundary conditions
            for node in self.bcs:
                x = self.xpts[node, 0] + disp[3 * node]
                y = self.xpts[node, 1] + disp[3 * node + 1]
                self.P_bc = ax.plot(x, y, "ko", ms=ms)

            # add color for the middle node
            for node in range(self.nnodes):
                if np.abs(xpts[node, 0]) < 1e-9 and np.abs(xpts[node, 1]) < 1e-9:
                    x = self.xpts[node, 0] + disp[3 * node]
                    y = self.xpts[node, 1] + disp[3 * node + 1]
                    self.P_h = ax.plot(x, y, "o", ms=ms, color="orange")

        # add the forces
        if quiver:
            for i in range(self.nnodes):
                x = self.xpts[i, 0] + disp[3 * i]
                y = self.xpts[i, 1] + disp[3 * i + 1]
                if self.forces[i, 0] != 0 or self.forces[i, 1] != 0:
                    a, b = 0, 0

                    if self.forces[i, 0] != 0:
                        a = 0.1 * self.L * np.sign(self.forces[i, 0])
                    if self.forces[i, 1] != 0:
                        b = 0.1 * self.L * np.sign(self.forces[i, 1])
                    if self.forces[i, 0] != 0 and self.forces[i, 1] != 0:
                        a, b = a / np.sqrt(2), b / np.sqrt(2)

                    self.P_force = ax.quiver(
                        x, y, a, b, color="r", scale=1.0, zorder=10, width=0.0075
                    )

        return

    def plot_domain(self, ax, u=None, x=None):

        self.visualize(ax, disp=u, x=x, domain=True)

        ax.text(
            0.5,
            0.95,
            r"Design Domain",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=title_fontsize,
        )

        ax.legend(
            [self.P_bc[0], self.P_h[0], self.P_x1[0], self.P_x2[0]],
            [
                "Boundary conditions",
                "Node applied relative \n displacement constraint",
                r"Design variable 1: beam area $x_{1}$",
                r"Design variable 2: beam area $x_{2}$",
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.2),
            fontsize=legend_fontsize,
            frameon=False,
            # # two columns
            # ncol=2,
        )

        return

    def plot_mode(self, gs, fig, Q, BLF, x, nfev, c=None):

        if nfev == 0:
            x_latex = r"$x^{*}_{i}$"
        elif nfev == 1:
            x_latex = r"$x^{*}_{ii}$"
        else:
            x_latex = r"$x^{*}_{iii}$"

        nn = 7
        N3GS1 = gs.subgridspec(1, nn)
        ax_mode = [fig.add_subplot(N3GS1[i]) for i in range(nn)]

        for i in range(nn):
            if i == 0:
                t = r"Optimized Design: " + x_latex
                self.visualize(ax_mode[i], x=x, c=c)
            else:
                t = (
                    r"Mode-${i}$".format(i=i)
                    + r": $BLF_{i}$ = ".format(i=i)
                    + f"{BLF[i-1]:.2f}"
                )
                self.visualize(ax_mode[i], disp=Q[:, i - 1], c=c)

            ax_mode[i].text(
                0.5,
                0.04,
                t,
                transform=ax_mode[i].transAxes,
                ha="center",
                va="center",
                fontsize=8,
            )
        # plt.subplots_adjust(wspace=0.2, hspace=-0.1)

        return


if __name__ == "__main__":
    # Set the random seed
    np.random.seed(0)

    # set the beam parameters
    settings = {
        "nelems": 12 * 6,
        "L": 1.0,
        "N": 10,
        "E": 1.0,
        "rg": 1.0,
        "rho": 1.0,
        "f": 1.0,
        "ks_rho_h": 10.0,
        "ks_rho_blf": 10.0,
    }

    # Create the domain, conn, xpts, bcs, forces
    domain = Domain(settings["nelems"], settings["L"], settings["f"])
    get_domain = {
        "1": domain.domain_bar,
        "2": domain.domain_square,
        "3": domain.domain_diamond,
        "4": domain.domain_circle,
        "5": domain.domain_square_mesh,
        "6": domain.domain_diamond_mesh,
    }
    conn, xpts, bcs, forces = get_domain["3"]()

    # Optimize the truss
    opt = Optimization(
        conn,
        xpts,
        bcs=bcs,
        forces=forces,
        E=settings["E"],
        rg=settings["rg"],
        N=settings["N"],
        ks_rho_h=settings["ks_rho_h"],
        ks_rho_blf=settings["ks_rho_blf"],
    )

    # # Optimize the truss
    h = [1e6, 0.23, 0.001]
    # res0, x_hist0 = opt.optimize(h_upper=h[0])
    # res1, x_hist1 = opt.optimize(h_upper=h[1])
    # res2, x_hist2 = opt.optimize(h_upper=h[2])
    # ic(res0, res1, res2)

    # # save the x results
    # np.save("x0.npy", res0.x)
    # np.save("x1.npy", res1.x)
    # np.save("x2.npy", res2.x)

    # read the x results
    x0 = np.load("x0.npy")
    x1 = np.load("x1.npy")
    x2 = np.load("x2.npy")

    # visualize the truss with optimized design
    with plt.style.context(["nature"]):
        fig = plt.figure(figsize=(9, 7), constrained_layout=True)  

        GS = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, height_ratios=[3, 4])
        N1GS1 = GS[0].subgridspec(1, 2, width_ratios=[2, 7])

        f1 = fig.add_subplot(N1GS1[0])
        domain.plot_domain(f1)

        nfev = 0
        for i, x in enumerate([x0, x1, x2]):
            gs = N1GS1[1].subgridspec(3, 1)[i]
            c = ["k", "r", "b"][i]
            x = opt.construst(x)

            BLF, Q = opt.solve_eigenvalue_problem(x)
            Q = Q / np.linalg.norm(Q, axis=0)
            domain.plot_mode(gs, fig, 2 * Q, BLF, x, nfev, c)

            ic(BLF[: settings["N"]])
            nfev += 1

        opt.plot_contour(GS[1], fig, x0, x1, x2, h[1], h[2], n=200)

        plt.savefig("simple_example0.png", bbox_inches="tight", dpi=800, pad_inches=0.0)

#################################################################
# cl = ax[i].clabel(
#     cg1,
#     fontsize=6,
#     inline_spacing=15.0,
#     use_clabeltext=True,
#     fmt="Mass constraint",
#     inline=False,
#     manual=[(0.3, 0.5)],
# )
# cl[0].set_position((0.3, 0.66))
# cg1.labelTexts[0].set_rotation(-45)

# # compute dx, dy and draw quiver
# m = 10
# msize = int(X.shape[0] / m)
# dx = np.zeros((msize, msize))
# dy = np.zeros((msize, msize))
# for i in range(msize):
#     for j in range(msize):
#         x = [X[i*m, j*m], Y[i*m, j*m]]
#         # use the finite difference to compute the gradient
#         dx[i, j] = (self.objective([x[0] + 1e-3, x[1]]) - self.objective(x)) / 1e-3
#         dy[i, j] = (self.objective([x[0], x[1] + 1e-3]) - self.objective(x)) / 1e-3

# X_new = X[::m, ::m][2:, 2:]
# Y_new = Y[::m, ::m][2:, 2:]
# dy = dy[2:, 2:]
# dx = dx[2:, 2:]
# Q = ax[0].quiver(X_new, Y_new, dx, dy, color="k", alpha=0.5, scale=10, scale_units="inches")
