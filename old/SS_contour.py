import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
import argparse
from paropt import ParOpt
import mpi4py.MPI as MPI
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch


class Truss:
    def __init__(self, q):
        self.q = q  # RAMP parameter

        # Create optimization problem instance
        problem = Problem(self)
        self.problem = problem

        return

    def compliance(self, x):
        """
        Compute the compliance objective s.t. design variable x
        f = [2, 1]
        """
        x1p = x[0] / (1 + self.q * (1 - x[0]))
        x2p = x[1] / (1 + self.q * (1 - x[1]))
        return 8 / (x1p + 5 * x2p) + 2 / (5 * x1p + x2p)

    def compliance_grad(self, x):
        """
        Compute the gradient analytically
        """
        grad = np.zeros(2)
        x1 = x[0]
        x2 = x[1]
        q = self.q

        grad[0] = (
            -2
            * (q * (x2 - 1) - 1) ** 2
            * (
                5 * (x1 * (q * (x2 - 1) - 1) + 5 * x2 * (q * (x1 - 1) - 1)) ** 2
                + 4 * (5 * x1 * (q * (x2 - 1) - 1) + x2 * (q * (x1 - 1) - 1)) ** 2
            )
            * (q * x1 - q * (x1 - 1) + 1)
            / (
                (x1 * (q * (x2 - 1) - 1) + 5 * x2 * (q * (x1 - 1) - 1)) ** 2
                * (5 * x1 * (q * (x2 - 1) - 1) + x2 * (q * (x1 - 1) - 1)) ** 2
            )
        )
        grad[1] = (
            -2
            * (q * (x1 - 1) - 1) ** 2
            * (
                (x1 * (q * (x2 - 1) - 1) + 5 * x2 * (q * (x1 - 1) - 1)) ** 2
                + 20 * (5 * x1 * (q * (x2 - 1) - 1) + x2 * (q * (x1 - 1) - 1)) ** 2
            )
            * (q * x2 - q * (x2 - 1) + 1)
            / (
                (x1 * (q * (x2 - 1) - 1) + 5 * x2 * (q * (x1 - 1) - 1)) ** 2
                * (5 * x1 * (q * (x2 - 1) - 1) + x2 * (q * (x1 - 1) - 1)) ** 2
            )
        )

        return grad

    def compliance_hess(self, x):
        """
        Compute the Hessian analytically
        """
        h = np.zeros((2, 2))
        x1 = x[0]
        x2 = x[1]
        q = self.q

        h[0, 0] = (
            4
            * (
                5
                * q
                * (q * x1 / (q * (x1 - 1) - 1) - 1)
                / (5 * x1 / (q * (x1 - 1) - 1) + x2 / (q * (x2 - 1) - 1)) ** 2
                + 4
                * q
                * (q * x1 / (q * (x1 - 1) - 1) - 1)
                / (x1 / (q * (x1 - 1) - 1) + 5 * x2 / (q * (x2 - 1) - 1)) ** 2
                - 25
                * (q * x1 / (q * (x1 - 1) - 1) - 1) ** 2
                / (5 * x1 / (q * (x1 - 1) - 1) + x2 / (q * (x2 - 1) - 1)) ** 3
                - 4
                * (q * x1 / (q * (x1 - 1) - 1) - 1) ** 2
                / (x1 / (q * (x1 - 1) - 1) + 5 * x2 / (q * (x2 - 1) - 1)) ** 3
            )
            / (q * (x1 - 1) - 1) ** 2
        )

        h[0, 1] = (
            -20
            * (q * (x1 - 1) - 1)
            * (q * (x2 - 1) - 1)
            * (
                (x1 * (q * (x2 - 1) - 1) + 5 * x2 * (q * (x1 - 1) - 1)) ** 3
                + 4 * (5 * x1 * (q * (x2 - 1) - 1) + x2 * (q * (x1 - 1) - 1)) ** 3
            )
            * (q * x1 - q * (x1 - 1) + 1)
            * (q * x2 - q * (x2 - 1) + 1)
            / (
                (x1 * (q * (x2 - 1) - 1) + 5 * x2 * (q * (x1 - 1) - 1)) ** 3
                * (5 * x1 * (q * (x2 - 1) - 1) + x2 * (q * (x1 - 1) - 1)) ** 3
            )
        )

        h[1, 0] = h[0, 1]
        h[1, 1] = (
            4
            * (
                q
                * (q * x2 / (q * (x2 - 1) - 1) - 1)
                / (5 * x1 / (q * (x1 - 1) - 1) + x2 / (q * (x2 - 1) - 1)) ** 2
                + 20
                * q
                * (q * x2 / (q * (x2 - 1) - 1) - 1)
                / (x1 / (q * (x1 - 1) - 1) + 5 * x2 / (q * (x2 - 1) - 1)) ** 2
                - (q * x2 / (q * (x2 - 1) - 1) - 1) ** 2
                / (5 * x1 / (q * (x1 - 1) - 1) + x2 / (q * (x2 - 1) - 1)) ** 3
                - 100
                * (q * x2 / (q * (x2 - 1) - 1) - 1) ** 2
                / (x1 / (q * (x1 - 1) - 1) + 5 * x2 / (q * (x2 - 1) - 1)) ** 3
            )
            / (q * (x2 - 1) - 1) ** 2
        )

        return h

    def optimize(self, maxit=5):
        # Create optimizer instance
        options = {
            "norm_type": "l1",
            "tr_max_iterations": maxit,
            "use_line_search": False,
        }

        opt = ParOpt.Optimizer(self.problem, options)
        opt.optimize()

        x_hist = self.problem.x_hist
        f_hist = self.problem.f_hist

        return x_hist[:-1], f_hist[:-1]  # Discard last point

    def check_gradients(self, h=1e-6):
        """
        check the grad and hess via finite difference
        """
        np.random.seed(0)  # Fix the seed
        self.q = np.random.rand()  # Reset a random (non-zero) q

        # Check gradients
        for i in range(5):
            x = np.random.rand(2)  # Create random designs
            grad_exact = self.compliance_grad(x)
            grad_fd = np.zeros(2)
            for i in range(2):
                p = np.zeros(2)
                p[i] = h
                grad_fd[i] = (self.compliance(x + p) - self.compliance(x)) / h
            err = grad_exact - grad_fd
            rel_err = np.linalg.norm(err) / np.linalg.norm(grad_exact)
            print("Gradient relative error: {:20.10e}".format(rel_err))

        # Check Hessians
        for i in range(5):
            x = np.random.rand(2)  # Create random designs
            hess_exact = self.compliance_hess(x)
            hess_fd = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    p = np.zeros(2)
                    q = np.zeros(2)
                    p[i] = h
                    q[j] = h
                    hess_fd[i, j] = (
                        self.compliance(x + p + q)
                        - self.compliance(x + p)
                        - self.compliance(x + q)
                        + self.compliance(x)
                    ) / h**2
            err = hess_exact - hess_fd
            rel_err = np.linalg.norm(err) / np.linalg.norm(hess_exact)
            print("Hessian relative error: {:20.10e}".format(rel_err))


class Problem(ParOpt.Problem):
    def __init__(self, truss):
        # Initialize the base class
        self.comm = MPI.COMM_WORLD
        self.nvars = 2
        self.ncon = 1
        super(Problem, self).__init__(self.comm, nvars=self.nvars, ncon=self.ncon)

        # Additional parameter(s)
        self.truss = truss

        # Save the design history
        self.x_hist = []
        self.f_hist = []

        return

    def getVarsAndBounds(self, x, lb, ub):
        # Init values
        x[0] = 0.6
        x[1] = 0.6

        # Bounds
        lb[:] = 1e-2
        ub[:] = 1.0
        return

    def evalObjCon(self, x):
        fobj = self.truss.compliance(x)
        con = [-x[0] - x[1] + 1.0]
        self.x_hist.append(np.array(x))
        self.f_hist.append(fobj)
        return 0, fobj, con

    def evalObjConGradient(self, x, g, A):
        g[0], g[1] = self.truss.compliance_grad(x)
        A[0][0] = -1.0
        A[0][1] = -1.0
        return 0


class Plotter:
    def __init__(self, xmin=0.1, xmax=1.0, resol=150, nlevel=50):
        self.xmin = xmin
        self.xmax = xmax
        self.resol = resol
        self.nlevel = nlevel  # number of levels in the contour plot

    def generate_contour(self, ax, q, maxit):
        # Set equal axes
        # ax.set_aspect('equal', adjustable='box')

        # Create analysis object, run optimization and parse paropt.tr
        truss = Truss(q=q)
        x_hist, f_hist = truss.optimize(maxit=maxit)

        with open("paropt.tr", "r") as f:
            lines = [l.strip() for l in f.readlines()][8:]  # Get history only
        skip_hist = []
        for line in lines:
            if "skipH" in line:
                skip_hist.append("red")
            else:
                skip_hist.append("blue")

        # Generate and populate meshgrid
        resol = self.resol
        xmin = self.xmin
        xmax = self.xmax
        X = np.linspace(xmin, xmax, resol)
        Y = np.linspace(xmin, xmax, resol)
        X, Y = np.meshgrid(X, Y)
        C = np.zeros(X.shape)  # compliance
        E = np.zeros(X.shape)  # curvature index

        for i in range(resol):
            for j in range(resol):
                x = [X[i, j], Y[i, j]]

                C[i, j] = truss.compliance(x)

                e, v = np.linalg.eigh(truss.compliance_hess(x))

                if e[0] < 0.0 and e[1] < 0.0:
                    E[i, j] = -1  # Negative definite: concave
                elif e[0] < 0.0 and e[1] > 0.0:
                    E[i, j] = 0  # Indefinite: saddle
                elif e[0] > 0.0:
                    E[i, j] = 1  # Positive definite: convex

        # Get value ranges
        C_min = np.min(C)
        C_max = np.max(C)
        E_min = np.min(E)
        E_max = np.max(E)

        nlevel = self.nlevel

        # Objective contour
        C_levels = np.logspace(np.log10(C_min), np.log10(C_max), num=nlevel)
        log_norm = colors.LogNorm(vmin=C_min, vmax=C_max)  # Use log-scale colormap
        comp_contour = ax.contour(
            X,
            Y,
            C,
            levels=C_levels,
            cmap="plasma",
            norm=log_norm,
            zorder=1,
            linewidths=1.0,
        )
        # Create colorbar
        divider = make_axes_locatable(ax)
        # comp_cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
        # comp_cbar = fig.colorbar(comp_contour, cax=comp_cbar_ax)

        # Curvature contour
        curv_contour = ax.contourf(
            X, Y, E, cmap="bwr_r", levels=[-1.5, -0.5, 0.5, 1.5], alpha=0.5, zorder=0
        )

        # Plot the constraint
        cx = [xmin, 1.0 - xmin]
        cy = [1.0 - xmin, xmin]
        ax.plot(
            cx, cy, color="black", linewidth=1.0, linestyle="--"
        )  # linewidth in pts

        # Append optimization history
        for i, x in enumerate(x_hist):
            ax.scatter(
                x[0],
                x[1],
                s=3**2,
                facecolors="white",
                linewidth=1.0,
                edgecolors=skip_hist[i],
                zorder=100,
                clip_on=False,
            )
            ax.annotate(i, (x[0] + 0.02, x[1] + 0.01))

        # Plot arrows
        for i in range(len(x_hist) - 1):
            pt1 = x_hist[i]
            pt2 = x_hist[i + 1]
            perp = pt2 - pt1
            perp = perp[::-1]
            perp[0] *= -1
            perp /= np.linalg.norm(perp)
            offset = -0.025
            arrow = FancyArrowPatch(
                pt1 + perp * offset,
                pt2 + perp * offset,
                arrowstyle="-|>",
                mutation_scale=5,
                shrinkA=3,
                shrinkB=3,
                linewidth=1.0,
                facecolor="white",
                edgecolor="black",
                zorder=150,
            )
            ax.add_artist(arrow)

        # Set labels
        ax.set_xlabel("$x_1$", labelpad=0.0, va="top")
        ax.set_ylabel("$x_2$", labelpad=2.0, va="bottom")  # labelpad in pts
        title = "q = {:.0f}".format(q)
        ax.set_title("$" + title + "$")

        # Manual legends
        blue_dot = Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.0,
            markersize=3,
            color="blue",
            label="$y^Ts > 0$",
        )
        red_dot = Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.0,
            markersize=3,
            color="red",
            label="$y^Ts < 0$",
        )
        legends = [blue_dot, red_dot]
        ax.legend(handles=legends, handlelength=1.0, framealpha=0.8, loc="upper right")

        # Manually set the ticks
        ax.set_xticks([1e-2, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # Debug info
        print(f"Generating contours..")
        for i, x in enumerate(x_hist):
            hess = truss.compliance_hess(x)
            e, v = np.linalg.eigh(hess)
            print(
                "[{:2d}] x:({:.3f}, {:.3f}) fobj: {:.5e}, mineig: {:.5e}".format(
                    i, x[0], x[1], f_hist[i], e[0]
                )
            )

        return curv_contour

    def generate_slice(self, ax, q, maxit):
        # Get parameters
        xmin = self.xmin
        xmax = self.xmax
        resol = self.resol

        # Create analysis object, run optimization and parse paropt.tr
        truss = Truss(q=q)
        x_hist, f_hist = truss.optimize(maxit=maxit)

        with open("paropt.tr", "r") as f:
            lines = [l.strip() for l in f.readlines()][8:]  # Get history only
        skip_hist = []
        for line in lines:
            if "skipH" in line:
                skip_hist.append("red")
            else:
                skip_hist.append("blue")

        # Allocate space
        x1 = np.linspace(xmin, xmax, resol)
        x2 = 1.0 - x1
        y = np.zeros(x1.shape)
        full_curv = np.zeros(x1.shape)
        redu_curv = np.zeros(x1.shape)

        # Evaluate compliance and curvature
        for i in range(resol):
            x = (x1[i], x2[i])
            y[i] = truss.compliance(x)

            hess = truss.compliance_hess(x)
            e, v = np.linalg.eigh(hess)
            p = np.array([-1.0, 1.0])
            redu_hess = p @ hess @ p  # H-inner product

            if e[0] < 0.0 and e[1] < 0.0:
                full_curv[i] = -1  # Negative definite: concave
            elif e[0] < 0.0 and e[1] > 0.0:
                full_curv[i] = 0  # Indefinite: saddle
            elif e[0] > 0.0:
                full_curv[i] = 1  # Positive definite: convex

            if redu_hess > 0:
                redu_curv[i] = 1.0
            else:
                redu_curv[i] = -1.0

        # Plot compliance
        ax.plot(x1, y, color="black", clip_on=False, linewidth=1.0, linestyle="--")

        # Create helper meshgrid to indicate reduced curvature
        ymin = min(y.min(), np.min(f_hist))
        _y = np.linspace(ymin, y.max(), 2)
        _X, _Y = np.meshgrid(x1, _y)
        _Z = np.zeros(_X.shape)

        for i in range(resol):
            # _Z[:, i] = full_curv[i]
            _Z[:, i] = redu_curv[i]

        # Scatter the design history
        for i, x in enumerate(x_hist):
            ax.scatter(
                x[0],
                f_hist[i],
                s=3**2,
                facecolors="white",
                linewidth=1.0,
                edgecolors=skip_hist[i],
                zorder=100,
                clip_on=False,
            )
            dx = 0.02 if i == 0 else -0.05
            dy = 0.1 if i == 0 else 0.2
            ax.annotate(i, (x[0] + dx, f_hist[i] + dy))

        # Plot arrows
        for i in range(len(x_hist) - 1):
            pt1 = np.array([x_hist[i][0], f_hist[i]])
            pt2 = np.array([x_hist[i + 1][0], f_hist[i + 1]])
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            skew = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
            perp = pt2 - pt1
            perp[1] /= skew
            perp = perp[::-1]
            perp[0] *= -1
            perp /= np.linalg.norm(perp)
            perp[1] *= skew
            offset = -0.02
            arrow = FancyArrowPatch(
                pt1 + perp * offset,
                pt2 + perp * offset,
                arrowstyle="-|>",
                mutation_scale=5,
                shrinkA=3,
                shrinkB=3,
                linewidth=1.0,
                facecolor="white",
                edgecolor="black",
                zorder=150,
            )
            ax.add_artist(arrow)

        # Fill regions
        curv_contour = ax.contourf(
            _X, _Y, _Z, cmap="PiYG", levels=[-1.5, -0.5, 0.5, 1.5], alpha=0.4, zorder=0
        )

        # Format figure
        ax.set_xlabel("$x_1$", labelpad=0.0, va="top")
        ax.set_ylabel("compliance", labelpad=2.0, va="bottom")  # labelpad in pts
        ax.set_ylim([ymin, y.max()])

        # Manual legends
        blue_dot = Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.0,
            markersize=3,
            color="blue",
            label="$y^Ts > 0$",
        )
        red_dot = Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.0,
            markersize=3,
            color="red",
            label="$y^Ts < 0$",
        )
        legends = [blue_dot, red_dot]
        ax.legend(handles=legends, handlelength=1.0, framealpha=0.8, loc="upper left")

        # Manually set the ticks
        ax.set_xticks([1e-2, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # Debug info
        print(f"Generating slices..")
        for i, x in enumerate(x_hist):
            hess = truss.compliance_hess(x)
            e, v = np.linalg.eigh(hess)
            print(
                "[{:2d}] x:({:.3f}, {:.3f}) fobj: {:.5e}, mineig: {:.5e}".format(
                    i, x[0], x[1], f_hist[i], e[0]
                )
            )

        return curv_contour


if __name__ == "__main__":
    # Take arguments
    p = argparse.ArgumentParser()
    p.add_argument("--q", default=[0.0, 2.0, 5.0], type=float, nargs="*")
    p.add_argument("--maxit", default=[4, 5, 5], type=int, nargs="*")
    p.add_argument("--nlevel", type=int, default=25)
    args = p.parse_args()

    # Make sure qvals and maxits are lists in same length
    qvals = args.q
    maxits = args.maxit
    if len(qvals) != len(maxits):
        raise RuntimeError("--q and --maxit should have same lengths")

    # Set up matplotlib
    mpl_style_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "smdoj.mplstyle"
    )
    plt.style.use(mpl_style_path)

    # Set up figure
    n = len(qvals)
    textwidth = 6.85  # LaTex \textwidth in inch
    columnwidth = 3.307  # LaTex \columnwidth in inch
    fig_width = 1.00 * textwidth
    fig_height = 0.9 * fig_width * 2 / n
    fig, axs = plt.subplots(nrows=2, ncols=n, figsize=(fig_width, fig_height))

    # Adjust paddings
    left = rcParams["figure.subplot.left"]
    right = rcParams["figure.subplot.right"]
    bottom = rcParams["figure.subplot.bottom"]
    top = rcParams["figure.subplot.top"]
    wspace = rcParams["figure.subplot.wspace"]
    hspace = rcParams["figure.subplot.hspace"]
    print(left, right, bottom, top)
    print(wspace, hspace)
    left = 0.05
    top = 0.95
    bottom = 0.06
    wspace = 0.2
    hspace = 0.2

    plt.subplots_adjust(
        left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace
    )

    # Create plotter for contour and slice plots
    plotter = Plotter(xmin=1e-2, nlevel=args.nlevel)

    # First row: contours
    for i, (ax, q, maxit) in enumerate(zip(axs[0], qvals, maxits)):
        curv_contour = plotter.generate_contour(ax, q, maxit)

        # Add the shared colorbar
        if i == n - 1:
            pos = ax.get_position(original=False)
            left, bottom, width, height = pos.bounds
            cax = plt.axes([left + width * 1.02, bottom, width * 0.05, height])
            curv_cbar = fig.colorbar(curv_contour, cax=cax)
            curv_cbar.set_ticks([-1.0, 0.0, 1.0])
            curv_cbar.set_ticklabels(
                [r"$H(x) \prec 0$", "indefinite\n$H(x)$", r"$H(x) \succ 0$"]
            )

    # Second row: slices
    for i, (ax, q, maxit) in enumerate(zip(axs[1], qvals, maxits)):
        curv_contour = plotter.generate_slice(ax, q, maxit)

        # Add the shared colorbar
        if i == n - 1:
            pos = ax.get_position(original=False)
            pos = ax.get_position(original=False)
            left, bottom, width, height = pos.bounds
            cax = plt.axes([left + width * 1.02, bottom, width * 0.05, height])
            curv_cbar = fig.colorbar(curv_contour, cax=cax)
            curv_cbar.set_ticks([-1.0, 0.0, 1.0])
            curv_cbar.set_ticklabels(
                [
                    "$p^TH(x)p$" + "\n" + "$< 0$",
                    "$p^TH(x)p$" + "\n" + "$= 0$",
                    "$p^TH(x)p$" + "\n$" + "> 0$",
                ]
            )

    # Save plot
    fig.savefig("contour.pdf")

    # Remove paropt history
    os.remove("paropt.out")
    os.remove("paropt.tr")
