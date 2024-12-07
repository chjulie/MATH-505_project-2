import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd
from mpi4py import MPI

from data_helpers import pol_decay, exp_decay
from functions import (
    create_sketch_matrix_gaussian_seq,
    create_sketch_matrix_SHRT_seq,
    is_power_of_two,
    rand_nystrom_parallel,
    create_sketch_matrix_gaussian_parallel,
    create_sketch_matrix_SHRT_parallel,
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

    # INITIALIZATION
    A = None
    AT = None
    Omega = None

    # GENERATE THE MATRIX A
    A_choice = "mnist"
    n = 1024
    n_local = int(n / n_blocks_row)

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

    if A_choice == "exp_decay" or A_choice == "pol_decay":
        l = 50
        Rs = [5, 10, 20]
        ks = [20, 20, 20]  # k < l!!!
        ps = [0.5, 1, 2]
        qs = [0.1, 0.25, 1.0]
        # for 1 matrix testing
        R = Rs[0]
        p = ps[2]
        k = ks[0]
        q = qs[2]

        # test: TODO remove
        # n = 16
        # n_local = int(n / n_blocks_row)
        # l = 5
        # R = 5
        # p = 0.5
        # k = 5
        # q = 0.1

        # generate at root and then broadcast
        if rank == 0:
            if A_choice == "exp_decay":
                A = exp_decay(n, R, q)
            else:
                A = pol_decay(n, R, p)
            arrs = np.split(A, n, axis=1)
            raveled = [np.ravel(arr) for arr in arrs]
            AT = np.concatenate(raveled)
            print("Shape of A: ", A.shape)
        AT = comm_rows.bcast(AT, root=0)
    elif A_choice == "mnist":
        l = 200
        k = 1

        if rank == 0:
            A = np.load("data/mnist_" + str(n) + ".npy")
            arrs = np.split(A, n, axis=1)
            raveled = [np.ravel(arr) for arr in arrs]
            AT = np.concatenate(raveled)
            print("Shape of A: ", A.shape)
        AT = comm_rows.bcast(AT, root=0)
    else:
        raise (NotImplementedError)
    # NOTE: TODO tecnnically no need to do transpose because we re using SPD matrices so A = AT

    # GENERATE THE MATRIX OMEGA
    sketching = "SHRT"  # "gaussian", "SHRT"

    # 1: SEQUENTIALLY
    # if rank == 0:
    #     # test if needed TODO: remove
    #     # Omega = np.arange(1, n * l + 1, 1, dtype=np.float64)
    #     # Omega = np.reshape(Omega, (n, l))
    #     if sketching == "gaussian":
    #         Omega = create_sketch_matrix_gaussian_seq(n, l)
    #     elif sketching == "SHRT":
    #         Omega = create_sketch_matrix_SHRT_seq(n, l)
    #     print("Shape of Omega: ", Omega.shape)
    # Omega = comm_rows.bcast(Omega, root=0)
    # Omega = comm_cols.bcast(Omega, root=0)

    # 2. IN PARALLEL
    # seed is SUPER important!!! (TODO: explain better)
    seed_global = 0
    Omega_local = np.empty((n_local, l), dtype=np.float64)
    OmegaT_local = np.empty((l, n_local), dtype=np.float64)
    if rank_rows == 0:  # Processors on 1st row of A
        if sketching == "gaussian":
            Omega_local = create_sketch_matrix_gaussian_parallel(
                n_local, l, seed=rank_cols
            )
        elif sketching == "SHRT":
            Omega_local = create_sketch_matrix_SHRT_parallel(
                n_local, l, seed_local=rank_cols, seed_global=seed_global
            )
        else:
            raise (NotImplementedError)

    if rank_cols == 0:  # Processors on 1st col of A
        if sketching == "gaussian":
            OmegaT_local = create_sketch_matrix_gaussian_parallel(
                n_local, l, seed=rank_rows
            ).T
        elif sketching == "SHRT":
            OmegaT_local = create_sketch_matrix_SHRT_parallel(
                n_local, l, seed_local=rank_rows, seed_global=seed_global
            ).T
        else:
            raise (NotImplementedError)

    # DISTRIBUTE THE MATRIX A TO GET A_local
    # Select columns, scatter them and put them in the right order
    submatrix = np.empty((n_local, n), dtype=np.float64)
    receiveMat = np.empty((n_local * n), dtype=np.float64)

    comm_cols.Scatterv(AT, receiveMat, root=0)
    subArrs = np.split(receiveMat, n_local)
    raveled = [np.ravel(arr, order="F") for arr in subArrs]
    submatrix = np.ravel(raveled, order="F")
    # Scatter the rows
    A_local = np.empty((n_local, n_local), dtype=np.float64)
    comm_rows.Scatterv(submatrix, A_local, root=0)

    # DISTRIBUTE OMEGA TO GET Omega_local and OmegaT_local
    # 1. SEQUENTIALLY
    # Omega_local = np.empty((n_local, l), dtype=np.float64)
    # comm_cols.Scatterv(Omega, Omega_local, root=0)

    # OmegaT_local = np.empty((n_local, l), dtype=np.float64)
    # comm_rows.Scatterv(Omega, OmegaT_local, root=0)
    # OmegaT_local = OmegaT_local.T

    # 2. IN PARALLEL
    Omega_local = comm_rows.bcast(Omega_local, root=0)  # broadcast to all rows
    OmegaT_local = comm_cols.bcast(OmegaT_local, root=0)  # broadcast to all columns

    # print(
    #     "Original rank: ",
    #     rank,
    #     "A local",
    #     A_local,
    #     "omega local: ",
    #     Omega_local,
    #     "omega transpose local: ",
    #     OmegaT_local,
    #     "\n",
    # )

    U_local, Sigma_2 = rand_nystrom_parallel(
        A_local,
        Omega_local,
        OmegaT_local,
        k,
        n,
        n_local,
        l,
        comm,
        comm_cols,
        comm_rows,
        rank,
        rank_cols,
        rank_rows,
        n_blocks_row,
    )

    U = np.empty((n, k), dtype=np.float64)

    comm_rows.Gather(U_local, U, root=0)

    if rank == 0:
        err = np.linalg.norm(U @ Sigma_2 @ U.T - A) / np.linalg.norm(A)
        print("Error in Froebenius norm: ", err)

        err_nuclear = np.linalg.norm(U @ Sigma_2 @ U.T - A, ord="nuc") / np.linalg.norm(
            A, ord="nuc"
        )
        print("Error in nuclear norm", err_nuclear)
        # print("Nuclear norm of A: ", np.linalg.norm(A, ord="nuc"))
        # print(
        #     "Nuclear norm of A_nystrom: ", np.linalg.norm(U @ Sigma_2 @ U.T, ord="nuc")
        # )
        # # Sanity check to make sure it is the same as above
        # print("Sum of diagonal entries nystrom: ", np.sum(np.diag(Sigma_2)))
