import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import time
import scipy

from data_helpers import pol_decay, exp_decay
from functions import (
    is_power_of_two,
    rand_nystrom_parallel,
    nuclear_error,
)
from stability_analysis_sequential import plot_errors

if __name__ == "__main__":


    # INITIALIZE MPI WORLD
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # comm.Barrier()

    if rank == 0:
        print(scipy.__version__)
        print(f" *** {rank} *** ")

    print(f"Rank: {rank}, Size: {size}")

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
    n = 1024
    n_local = int(n / n_blocks_row)
    As = []

    # Parameters for the polynomial and exponential matrices
    R = 10
    ps = [0.5, 1, 2]
    qs = [0.1, 0.25, 1.0]

    # Parameters to vary for stability analysis
    ls = [150, 200, 250, 500]
    ks = [5, 10, 25, 50, 100, 150]

    # Generate the matrices
    if rank == 0:
        for p in ps:
            As.append(pol_decay(n, R, p))
        for q in qs:
            As.append(exp_decay(n, R, q))

    if rank == 0:
        print(" > Matrices initialized")
    seed_global = 42

    errors_gaussian_all = []
    errors_SHRT_all = []

    for i, A in enumerate(As):
        if rank == 0:
            print(f"Matrix {i}")
        errors_gaussian = []
        errors_SHRT = []

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

        # Initialization
        if rank == 0:
            AT = A 
        else:
            A = None
            AT = None
        
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

        for l in ls:
            if rank == 0:
                print(f" > l = {l}")

            errors_gaussian_tmp = []
            errors_SHRT_tmp = []

            for k in ks:
                if rank == 0:
                    print(f"  > k = {k}")

                # gaussian sketching matrix
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
                    print_computation_times=False
                )
                U = None
                if rank == 0:
                    U = np.empty((n, k), dtype=np.float64)
                if rank_rows == 0:
                    comm_cols.Gather(U_local, U, root=0)

                if rank == 0:
                    errors_gaussian_tmp.append(nuclear_error(A, U, Sigma_2))

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
                    print_computation_times=False
                )
                U = None
                if rank == 0:
                    U = np.empty((n, k), dtype=np.float64)
                if rank_rows == 0:
                    comm_cols.Gather(U_local, U, root=0)

                if rank == 0:
                    errors_gaussian_tmp.append(nuclear_error(A, U, Sigma_2))

            if rank == 0:
                errors_gaussian.append(errors_gaussian_tmp)
                errors_SHRT.append(errors_SHRT_tmp)

        if rank == 0:
            errors_gaussian_all.append(errors_gaussian)
            errors_SHRT_all.append(errors_SHRT)

    if rank == 0:
        print(f" > Computations done!")
    # Plot for each matrix and method
    results_folder = "results/numerical_stability_parallel_64"
    if rank == 0:
        for i in range(len(As)):
            # Gaussian method
            plot_errors(errors_gaussian_all, "Gaussian", results_folder, ks, ls, i)

            # SHRT method
            plot_errors(errors_SHRT_all, "SHRT", results_folder, ks, ls, i)

        print(" > Program finished successfully!")