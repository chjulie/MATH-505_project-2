import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('ggplot')
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_helpers import pol_decay, exp_decay
from functions import (
    is_power_of_two,
    rand_nystrom_parallel,
    rand_nystrom_sequential,
    nuclear_error_relative,
    plot_errors,
)

if __name__ == "__main__":

    # INITIALIZATION
    n = 1024
    As = []
    titles = []

    # Parameters for the polynomial and exponential matrices
    R = 10
    ps = [0.5, 1, 2]
    qs = [0.1, 0.25, 1.0]

    # Parameters to vary for stability analysis
    ls = [150, 200, 250, 500]
    colors = ["#0b3954", "#087e8b", "#ff5a5f", "#c81d25"]
    ks = [5, 10, 25, 50, 100, 150]

    # Generate the matrices
    for p in ps:
        As.append(pol_decay(n, R, p))
        titles.append(r"$p=$" + str(p))
    for q in qs:
        As.append(exp_decay(n, R, q))
        titles.append(r"$q=$" + str(q))

    seed_sequential = 5

    errors_gaussian_all = []
    errors_SHRT_all = []

    for i, A in enumerate(As):
        print(i + 1, "th matrix")
        errors_gaussian = []
        errors_SHRT = []

        for l in ls:
            errors_gaussian_tmp = []
            errors_SHRT_tmp = []
            for k in ks:

                # Gaussian sketching matrix
                U, Sigma_2 = rand_nystrom_sequential(
                    A=A,
                    seed=seed_sequential,
                    n=n,
                    sketching="gaussian",
                    k=k,
                    l=l,
                    return_extra=False,
                    return_runtimes=False,
                    print_computation_times=False,
                )
                errors_gaussian_tmp.append(nuclear_error_relative(A, U, Sigma_2))

                # SHRT sketching matrix
                U, Sigma_2 = rand_nystrom_sequential(
                    A=A,
                    seed=seed_sequential,
                    n=n,
                    sketching="SHRT",
                    k=k,
                    l=l,
                    return_extra=False,
                    return_runtimes=False,
                    print_computation_times=False,
                )
                errors_SHRT_tmp.append(nuclear_error_relative(A, U, Sigma_2))

            errors_gaussian.append(errors_gaussian_tmp)
            errors_SHRT.append(errors_SHRT_tmp)

        errors_gaussian_all.append(errors_gaussian)
        errors_SHRT_all.append(errors_SHRT)

    # print("Gaussian sketching matrix: ", errors_gaussian_all)
    # print("SHRT sketching matrix: ", errors_SHRT_all)

    # Plot for each matrix and method
    results_folder = "results/numerical_stability_sequential"
    for i in range(len(As)):
        # Gaussian method
        plot_errors(
            errors_gaussian_all,
            "Gaussian",
            results_folder,
            ks,
            ls,
            i,
            colors,
            titles[i],
        )

        # SHRT method
        plot_errors(
            errors_SHRT_all, "SHRT", results_folder, ks, ls, i, colors, titles[i]
        )
