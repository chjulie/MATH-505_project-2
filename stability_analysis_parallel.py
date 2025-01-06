import numpy as np
from mpi4py import MPI
import json

from data_helpers import pol_decay
from functions import (
    is_power_of_two,
    rand_nystrom_parallel,
    nuclear_error_relative,
)

if __name__ == "__main__":

    # INITIALIZE MPI WORLD
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_blocks_row = int(np.sqrt(size))
    n_blocks_col = int(np.sqrt(size))

    if n_blocks_col**2 != int(size):
        if rank == 0:
            print("The number of processors must be a perfect square")
        exit(-1)
    if not is_power_of_two(n_blocks_row):
        if rank == 0:
            print(
                "The square root of the number of processors should be a power of 2 (for TSQR)"
            )
        exit(-1)

    comm_cols = comm.Split(color=rank / n_blocks_row, key=rank % n_blocks_row)
    comm_rows = comm.Split(color=rank % n_blocks_row, key=rank / n_blocks_row)

    # Get ranks of subcommunicator
    rank_cols = comm_cols.Get_rank()
    rank_rows = comm_rows.Get_rank()
    if rank == 0:
        print(" > MPI initialized")

    # INITIALIZATION OF MATRICES
    n = 4096
    n_local = int(n / n_blocks_row)

    # Parameters to vary for stability analysis
    l = 64
    ks = [10, 25, 32, 64]

    # Generate the matrix
    if rank == 0:
        A = pol_decay(n, R=10, p=2)
    else:
        A = None

    if rank == 0:
        print(" > Matrix initialized")

    seed_global = 42

    # DISTRIBUTE A OVER PROCESSORS
    # check the size of A
    if n_blocks_col * n_local != n:  # Check n is divisible by n_blocks_row
        if rank == 0:
            print(
                "n should be divisible by sqrt(P) where P is the number of processors"
            )
        exit(-1)
    if not is_power_of_two(n):  # Check n is a power of 2
        if rank == 0:
            print("n should be a power of 2")
        exit(-1)

    # Distribute A
    AT = None
    if rank == 0:
        AT = A
    AT = comm_rows.bcast(AT, root=0)

    submatrix = np.empty((n_local, n), dtype=np.float64)
    receiveMat = np.empty((n_local * n), dtype=np.float64)

    comm_cols.Scatterv(AT, receiveMat, root=0)
    subArrs = np.split(receiveMat, n_local)
    raveled = [np.ravel(arr, order="F") for arr in subArrs]
    submatrix = np.ravel(raveled, order="F")
    # Scatter the rows
    A_local = np.empty((n_local, n_local), dtype=np.float64)
    comm_rows.Scatterv(submatrix, A_local, root=0)

    if rank == 0:
        print(f" > l = {l}")

    errors_gaussian = []
    errors_SHRT = []

    for k in ks:
        if k <= l:
            if rank == 0:
                print(f"  > k = {k}")

            # Gaussian sketching matrix
            U_local, Sigma_2 = rand_nystrom_parallel(
                A_local=A_local,
                seed_global=seed_global,
                n=n,
                k=k,
                n_local=n_local,
                l=l,
                sketching="gaussian",
                comm=comm,
                comm_cols=comm_cols,
                comm_rows=comm_rows,
                rank=rank,
                rank_cols=rank_cols,
                rank_rows=rank_rows,
                size_cols=comm_cols.Get_size(),
                print_computation_times=False,
                return_runtimes=False,
            )
            U = None
            if rank == 0:
                U = np.empty((n, k), dtype=np.float64)
            if rank_rows == 0:
                comm_cols.Gather(U_local, U, root=0)

            if rank == 0:
                errors_gaussian.append(nuclear_error_relative(A, U, Sigma_2))

            # SHRT sketching matrix
            U_local, Sigma_2 = rand_nystrom_parallel(
                A_local=A_local,
                seed_global=seed_global,
                n=n,
                k=k,
                n_local=n_local,
                l=l,
                sketching="SHRT",
                comm=comm,
                comm_cols=comm_cols,
                comm_rows=comm_rows,
                rank=rank,
                rank_cols=rank_cols,
                rank_rows=rank_rows,
                size_cols=comm_cols.Get_size(),
                print_computation_times=False,
                return_runtimes=False,
            )
            U = None
            if rank == 0:
                U = np.empty((n, k), dtype=np.float64)
            if rank_rows == 0:
                comm_cols.Gather(U_local, U, root=0)

            if rank == 0:
                errors_SHRT.append(nuclear_error_relative(A, U, Sigma_2))

    if rank == 0:
        print(f" > Computations done!")
        # Save results in a JSON file
        json_file = (
            "results/numerical_stability_data/P" + str(size) + "_n" + str(n) + ".json"
        )
        results = {}
        results["gaussian"] = errors_gaussian
        results["SHRT"] = errors_gaussian
        info = {}
        info["k"] = ks
        info["l"] = l

        data = {"results": results, "info": info}

        with open(json_file, "w") as file:
            json.dump(data, file, indent=4)
