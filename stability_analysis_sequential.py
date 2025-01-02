import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from data_helpers import pol_decay, exp_decay
from functions import (
    is_power_of_two,
    rand_nystrom_parallel,
    rand_nystrom_sequential,
    nuclear_error,
)


def plot_errors(errors_all, method_name, results_folder, ks, ls, matrix_index):
    """
    Generate a plot for errors as a function of k for each matrix and method.

    Parameters:
    - errors_all: List of errors for the sketching method.
    - method_name: String, either "Gaussian" or "SHRT".
    - ks: List of k values.
    - ls: List of l values.
    - matrix_index: Index of the current matrix.
    """
    errors = np.array(errors_all[matrix_index])
    plt.figure(figsize=(10, 6))
    for l_idx, l_value in enumerate(ls):
        errors_l = errors[l_idx, :]
        plt.plot(ks, errors_l, label=f"l={l_value}", marker="o")

    plt.yscale("log")  # Set y-axis to logarithmic scale
    plt.title(f"Matrix {matrix_index + 1}: {method_name} Sketching")
    plt.xlabel("k")
    plt.ylabel("Nuclear Norm Error")
    plt.legend()
    plt.grid(True)
    plt.savefig(results_folder + "/" + str(matrix_index) + "_" + method_name + ".png")


if __name__ == "__main__":

    # INITIALIZATION
    n = 1024
    As = []

    # Parameters for the polynomial and exponential matrices
    R = 10
    ps = [0.5, 1, 2]
    qs = [0.1, 0.25, 1.0]

    # Parameters to vary for stability analysis
    ls = [150, 200, 250, 500]
    ks = [5, 10, 25, 50, 100, 150]

    # Generate the matrices
    for p in ps:
        As.append(pol_decay(n, R, p))
    for q in qs:
        As.append(exp_decay(n, R, q))

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
                errors_gaussian_tmp.append(nuclear_error(A, U, Sigma_2))

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
                errors_SHRT_tmp.append(nuclear_error(A, U, Sigma_2))

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
        plot_errors(errors_gaussian_all, "Gaussian", results_folder, ks, ls, i)

        # SHRT method
        plot_errors(errors_SHRT_all, "SHRT", results_folder, ks, ls, i)
