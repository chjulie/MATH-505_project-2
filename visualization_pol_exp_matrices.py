import numpy as np
import matplotlib.pyplot as plt

# plt.style.use("ggplot")

from data_helpers import pol_decay, exp_decay

if __name__ == "__main__":

    # INITIALIZATION
    n = 1024
    As_pol = []
    As_exp = []

    # Parameters for the polynomial and exponential matrices
    R = 10
    ps = [0.5, 1, 2]
    qs = [0.1, 0.25, 1.0]

    # Parameters to vary for stability analysis
    ls = [150, 200, 250, 500]
    ks = [5, 10, 25, 50, 100, 150]

    # Generate the matrices
    for p in ps:
        As_pol.append(pol_decay(n, R, p))
    for q in qs:
        As_exp.append(exp_decay(n, R, q))

    fig_diag, axs_diag = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    # fig_diag.suptitle("Diagonal entries of A")
    axs_diag[0].set_title(r"$A_1$: Polynomial Decay")
    axs_diag[1].set_title(r"$A_2$: Exponential Decay")

    colors_pol = ["#0b3954", "#087e8b", "#bfd7ea"]
    colors_exp = ["#c81d25", "#ff5a5f", "#ffa3a5"]

    for i, A in enumerate(As_pol):
        axs_diag[0].loglog(
            np.arange(n), np.diag(A), c=colors_pol[i], label=r"$p=$" + str(ps[i])
        )
        axs_diag[0].legend(loc="upper left")

    for i, A in enumerate(As_exp):
        axs_diag[1].loglog(
            np.arange(n), np.diag(A), c=colors_exp[i], label=r"$q=$" + str(qs[i])
        )
        axs_diag[1].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("results/matrix_visualization/vis_matrices.png", bbox_inches="tight")
    plt.savefig(
        "results/matrix_visualization/vis_matrices.svg",
        format="svg",
        bbox_inches="tight",
    )
