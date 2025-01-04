import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import time

from data_helpers import pol_decay, exp_decay
from functions import (
    is_power_of_two,
    rand_nystrom_parallel,
    nuclear_error_relative,
    nuclear_error,
    plot_errors,
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
    n = 2048
    n_local = int(n / n_blocks_row)
    As = []
    titles = []

    # Parameters for the polynomial and exponential matrices
    R = 10
    ps = [0.5, 1, 2]
    qs = [0.1, 0.25, 1.0]

    # Parameters to vary for stability analysis
    ls = [50]  # , 150, 250, 500, 700]
    colors = ["#0b3954", "#087e8b", "#bfd7ea", "#ff5a5f", "#c81d25"]
    ks = [5, 10, 25, 50, 100, 150, 200, 300]

    # Generate the matrices
    for p in ps:
        if rank == 0:
            As.append(pol_decay(n, R, p))
            titles.append(r"$p=$" + str(p))
        else:
            As.append(None)
            titles.append(None)
    for q in qs:
        if rank == 0:
            As.append(exp_decay(n, R, q))
            titles.append(r"$q=$" + str(q))
        else:
            As.append(None)
            titles.append(None)

    if rank == 0:
        print(" > Matrices initialized")

    # calculate the sum of k+1 eigenvalues
    sum_eigenvalues = np.zeros((len(As), len(ks)))
    if rank == 0:
        for i, A in enumerate(As):
            for j, k in enumerate(ks):
                sum_eigenvalues[i, j] = sum(np.diag(A)[k:])

    seed_global = 0  # 42

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

        for l in ls:
            if rank == 0:
                print(f" > l = {l}")

            errors_gaussian_tmp = []
            errors_SHRT_tmp = []

            for k in ks:
                if k <= l:
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
                        print_computation_times=False,
                        return_runtimes=False,
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
                        print_computation_times=False,
                        return_runtimes=False,
                    )
                    U = None
                    if rank == 0:
                        U = np.empty((n, k), dtype=np.float64)
                    if rank_rows == 0:
                        comm_cols.Gather(U_local, U, root=0)

                    if rank == 0:
                        errors_SHRT_tmp.append(nuclear_error(A, U, Sigma_2))

            if rank == 0:
                errors_gaussian.append(errors_gaussian_tmp)
                errors_SHRT.append(errors_SHRT_tmp)

        if rank == 0:
            errors_gaussian_all.append(errors_gaussian)
            errors_SHRT_all.append(errors_SHRT)

    if rank == 0:
        print(f" > Computations done!")

    # Plot for each matrix and method
    results_folder = "results/numerical_stability_parallel"
    if rank == 0:
        for i in range(len(As)):
            # Gaussian method
            plot_errors(
                errors_gaussian_all[i],
                "gaussian",
                results_folder,
                ks,
                ls,
                i,
                colors,
                titles[i],
                optimal_error=sum_eigenvalues[i],
                y_label="nuclear norm error",
            )

            # SHRT method
            plot_errors(
                errors_SHRT_all[i],
                "SHRT",
                results_folder,
                ks,
                ls,
                i,
                colors,
                titles[i],
                optimal_error=sum_eigenvalues[i],
                y_label="nuclear norm error",
            )

        print(" > Program finished successfully!")

    finish_timestamp = time.localtime(time.time())
    formatted_time = time.strftime("%H:%M:%S", finish_timestamp)
    print(f" * proc {rank}: finished program at {formatted_time} ")
