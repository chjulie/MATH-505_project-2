import numpy as np
import matplotlib.pyplot as plt
from data_helpers import pol_decay, exp_decay
from functions import rand_nystrom_seq
from plot import setIndividualTitles, setColNames

if __name__ == "__main__":

    n = 10**3
    l = 50
    Rs = [5, 10, 20]
    ks = [20, 20, 20]  # k < l!!!
    ps = [0.5, 1, 2]
    qs = [0.1, 0.25, 1.0]
    Omega = np.random.normal(loc=0.0, scale=1.0, size=[n, l])
    colors = ["#003aff", "#ff8f00", "#b200ff"]

    # Grid of plots
    fig_diag, axs_diag = plt.subplots(nrows=3, ncols=3, figsize=(16, 8))
    fig_diag.suptitle("Diagonal entries of A")
    fig_err, axs_err = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
    fig_err.suptitle("Relative error of approximation")
    fig_condB, axs_condB = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
    fig_condB.suptitle("Condition number of B")
    fig_condA, axs_condA = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
    fig_condA.suptitle("Condition number of A")

    for i in range(3):
        # Iterate through the Rs
        R = Rs[i]
        k = ks[i]
        err_mnist = 3 * [0]
        err_poly = 3 * [0]
        err_exp = 3 * [0]
        conB_mnist = 3 * [0]
        conB_poly = 3 * [0]
        conB_exp = 3 * [0]
        conA_mnist = 3 * [0]
        conA_poly = 3 * [0]
        conA_exp = 3 * [0]
        for j in range(3):
            # Iterate through the parameters of the matrices
            p = ps[j]
            q = qs[j]
            ## A_mnist =
            A_poly = pol_decay(n, R, p)
            A_exp = exp_decay(n, R, q)
            # Loglog the diagonals
            # axs_diag[i, 0].loglog(
            #     np.arange(n), np.diag(A_mnist), c=colors[j], label=r"$\xi=$" + str(xi)
            # )
            # axs_diag[i, 0].legend(loc="upper left")
            axs_diag[i, 1].loglog(
                np.arange(n), np.diag(A_poly), c=colors[j], label=r"$p=$" + str(p)
            )
            axs_diag[i, 1].legend(loc="upper left")
            axs_diag[i, 2].loglog(
                np.arange(n), np.diag(A_exp), c=colors[j], label=r"$q=$" + str(q)
            )
            axs_diag[i, 2].legend(loc="upper left")
            # Randomized Nystrom
            # U_mnist, S2_mnist, UT_mnist, S_B_mnist, rankA_mnist = rand_nystrom_seq(
            #     A_mnist, Omega
            # )
            U_poly, S2_poly, UT_poly, S_B_poly, rankA_poly = rand_nystrom_seq(
                A_poly, Omega, k
            )
            U_exp, S2_exp, UT_exp, S_B_exp, rankA_exp = rand_nystrom_seq(
                A_exp, Omega, k
            )
            # Save errors to loglog
            # err_mnist[j] = np.linalg.norm(U_mnist @ S2_mnist @ UT_mnist - A_mnist) / np.linalg.norm(A_mnist)
            err_poly[j] = np.linalg.norm(
                U_poly @ S2_poly @ UT_poly - A_poly
            ) / np.linalg.norm(A_poly)
            err_exp[j] = np.linalg.norm(
                U_exp @ S2_exp @ UT_exp - A_exp
            ) / np.linalg.norm(A_poly)
            # conB_mnist[j] = S_B_mnist
            conB_poly[j] = S_B_poly
            conB_exp[j] = S_B_exp
            # conA_mnist[j] = np.linalg.cond(A_mnist)
            conA_poly[j] = np.linalg.cond(A_poly)
            conA_exp[j] = np.linalg.cond(A_exp)
        # Loglog

        # axs_err[0].loglog(??, err_mnist, c=colors[i], label="R= " + str(R))
        axs_err[1].loglog(
            ps, err_poly, c=colors[i], label="R= " + str(R) + " | k = " + str(k)
        )
        axs_err[2].loglog(
            qs, err_exp, c=colors[i], label="R= " + str(R) + " | k = " + str(k)
        )
        # axs_condB[0].loglog(??, conB_mnist, c=colors[i], label="R= " + str(R))
        axs_condB[1].loglog(
            ps, conB_poly, c=colors[i], label="R= " + str(R) + " | k = " + str(k)
        )
        axs_condB[2].loglog(
            qs, conB_exp, c=colors[i], label="R= " + str(R) + " | k = " + str(k)
        )
        # axs_condA[0].loglog(??, conA_mnist, c=colors[i], label="R= " + str(R))
        axs_condA[1].loglog(
            ps, conA_poly, c=colors[i], label="R= " + str(R) + " | k = " + str(k)
        )
        axs_condA[2].loglog(
            qs, conA_exp, c=colors[i], label="R= " + str(R) + " | k = " + str(k)
        )

    # Add legends and format titles
    # axs_err[0].legend(loc="upper left")
    axs_err[1].legend(loc="upper left")
    axs_err[2].legend(loc="upper left")
    # axs_condB[0].legend(loc="upper left")
    axs_condB[1].legend(loc="upper left")
    axs_condB[2].legend(loc="upper left")
    # axs_condA[0].legend(loc="upper left")
    axs_condA[1].legend(loc="upper left")
    axs_condA[2].legend(loc="upper left")

    setColNames(axs_diag)
    setColNames(axs_err, cols=[r"$\xi$", "p", "q"], rows=["Rel err"])
    setIndividualTitles(axs_err)
    setColNames(axs_condB, cols=[r"$\xi$", "p", "q"], rows=[r"$\kappa$"])
    setIndividualTitles(axs_condB)
    setColNames(axs_condA, cols=[r"$\xi$", "p", "q"], rows=[r"$\kappa$"])
    setIndividualTitles(axs_condA)

    # Save the figures
    fig_diag.savefig(
        "results/fig_diag.png", dpi=300, bbox_inches="tight"
    )  # Save diagonal entries figure
    fig_err.savefig(
        "results/fig_err.png", dpi=300, bbox_inches="tight"
    )  # Save relative error figure
    fig_condB.savefig(
        "results/fig_condB.png", dpi=300, bbox_inches="tight"
    )  # Save condition number B figure
    fig_condA.savefig(
        "results/fig_condA.png", dpi=300, bbox_inches="tight"
    )  # Save condition number A figure
