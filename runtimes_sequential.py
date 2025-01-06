import numpy as np
import json

from data_helpers import pol_decay, exp_decay
from functions import (
    rand_nystrom_sequential,
)

if __name__ == "__main__":

    # INITIALIZE DATA STORAGE
    json_file = "results/runtimes_1.json"
    data = []

    # LOOP OVER MATRIX SIZES
    ns = [8192]  # [1024, 2048, 4096, 8192]

    seed_sequential = 42

    for n in ns:
        print(f" > n = {n}")

        # INITIALIZATION
        A = None

        A_choice = "mnist"
        seed_global = 42
        l = 128
        k = 100  # k <=l !! + does not influence runtime

        # GENERATE THE MATRIX A

        if A_choice == "exp_decay" or A_choice == "pol_decay":
            R = 10
            if A_choice == "exp_decay":
                q = 0.1  # Change if needed
                A = exp_decay(n, R, q)
            elif A_choice == "pol_decay":
                p = 0.5  # Change if needed
                A = pol_decay(n, R, p)
            else:
                raise (NotImplementedError)

        elif A_choice == "mnist":
            A = np.load("data/mnist_" + str(n) + ".npy")

        else:
            raise (NotImplementedError)

        print("Shape of A: ", A.shape)

        it = 5
        average_runtimes_gaussian = np.array(np.zeros(5))
        average_runtimes_SHRT = np.array(np.zeros(5))

        # Gaussian sketching matrix
        print("  > gaussian sketching")

        for i in range(it):
            _, _, runtimes = rand_nystrom_sequential(
                A=A,
                seed=seed_sequential,
                n=n,
                sketching="gaussian",
                k=k,  # Truncation rank
                l=l,
                return_extra=False,
                return_runtimes=True,
                print_computation_times=False,
            )

            average_runtimes_gaussian += runtimes
            print(f"   > it {i+1} done")

        average_runtimes_gaussian = average_runtimes_gaussian / it

        # SHRT sketching matrix
        print("  > SHRT sketching matrix")

        for i in range(it):
            _, _, runtimes = rand_nystrom_sequential(
                A=A,
                seed=seed_sequential,
                n=n,
                sketching="SHRT",
                k=k,  # Truncation rank
                l=l,
                return_extra=False,
                return_runtimes=True,
                print_computation_times=False,
            )

            average_runtimes_SHRT += runtimes
            print(f"   > it {i+1} done")

        average_runtimes_SHRT = average_runtimes_SHRT / it

        run_details = {
            "matrix_size": n,
            "n_proc": 1,
            "runtimes_gaussian": average_runtimes_gaussian.tolist(),
            "runtimes_SHRT": average_runtimes_SHRT.tolist(),
        }
        data.append(run_details)

    with open(json_file, "w") as file:
        json.dump(data, file, indent=4)

    print(" > Program finished successfully!")
