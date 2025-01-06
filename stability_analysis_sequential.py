import numpy as np
import json

from data_helpers import pol_decay
from functions import (
    rand_nystrom_sequential,
    nuclear_error_relative,
)

if __name__ == "__main__":

    # INITIALIZATION
    n = 4096

    # Parameters to vary for stability analysis
    l = 64
    ks = [10, 25, 32, 64]

    # Generate the matrix
    A = pol_decay(n, R=10, p=2)

    seed_sequential = 42

    errors_gaussian_all = []
    errors_SHRT_all = []

    print(f" > l = {l}")

    errors_gaussian = []
    errors_SHRT = []

    for k in ks:
        if k <= l:
            print(f"  > k = {k}")

            U, Sigma_2 = rand_nystrom_sequential(
                A=A,
                seed=seed_sequential,
                n=n,
                sketching="gaussian",
                k=k,  # Truncation rank
                l=l,
                return_extra=False,
                return_runtimes=False,
                print_computation_times=False,
            )

            errors_gaussian.append(nuclear_error_relative(A, U, Sigma_2))

            U, Sigma_2 = rand_nystrom_sequential(
                A=A,
                seed=seed_sequential,
                n=n,
                sketching="SHRT",
                k=k,  # Truncation rank
                l=l,
                return_extra=False,
                return_runtimes=False,
                print_computation_times=False,
            )

            errors_SHRT.append(nuclear_error_relative(A, U, Sigma_2))

    print(f" > Computations done!")
    # Save results in a JSON file
    json_file = "results/numerical_stability_data/P1_n" + str(n) + ".json"
    results = {}
    results["gaussian"] = errors_gaussian
    results["SHRT"] = errors_gaussian
    info = {}
    info["k"] = ks
    info["l"] = l

    data = {"results": results, "info": info}

    with open(json_file, "w") as file:
        json.dump(data, file, indent=4)
