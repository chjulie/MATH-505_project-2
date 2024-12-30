import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd
from mpi4py import MPI
import random
import time

from data_helpers import pol_decay, exp_decay
from functions import (
    create_sketch_matrix_gaussian_seq,
    create_sketch_matrix_SHRT_seq,
    is_power_of_two,
    rand_nystrom_parallel_SHRT,
    SFWHT,
    FWHT,
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

    # print('rank cols: ', rank_cols)
    # print('rank rows: ', rank_rows)

    # INITIALIZATION
    A = None
    AT = None
    Omega = None

    # GENERATE THE MATRIX A
    A_choice = "mnist"
    n = 8#8192
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
        l = 2#200
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
    # NOTE: tecnnically no need to do transpose because we re using SPD matrices so A = AT

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

    # print(f" * Rank {rank}, rank_cols: {rank_cols}, rank_rows: {rank_rows}: A_local: {A_local}")

    # CHOOSE SKETCHING MATRIX OMEGA
    sketching = "SHRT"  # "gaussian", "SHRT"

    # 2. IN PARALLEL
    # seed is SUPER important!!! (TODO: explain better)
    seed_global = 0

    # NOTE: check with Mathilde
    # seed_local = rank_rows
    # if rank_cols == 0:
    #     seed_local = rank_rows

    # Broadcast the local seed across rows
    # NOTE: to broadcast across rows i use comm_cols??
    # seed_local = comm_cols.bcast(seed_local, root=0)
    # print(f" * Rank {rank}, rank_cols: {rank_cols}, rank_rows: {rank_rows}: seed_local = {seed_local}")

    # NOTE: not necessary
    # elif rank_rows == 0:
    #     seed_local = int(time.time()) + 42
        # seed_local = rank_cols

    U_local, Sigma_2 = rand_nystrom_parallel_SHRT(
        A_local = A_local,
        seed_global = seed_global,
        k = k, # TODO: check where
        n = n,
        n_local = n_local,
        l = l, # TODO: check where
        sketching = sketching, 
        comm = comm,
        comm_cols = comm_cols,
        comm_rows = comm_rows,
        rank = rank,
        rank_cols = rank_cols,
        rank_rows = rank_rows,
        size_cols = comm_cols.Get_size(),
    )

    if rank == 0:
        print(f" * U_local.shape:  {U_local.shape}")
        print(f" * Sigma_2.shape:  {Sigma_2.shape}")

    finish_timestamp = time.localtime(time.time())
    formatted_time = time.strftime("%H:%M:%S", finish_timestamp)
    print(f" ** proc {rank}: finished program at {formatted_time} ** ")
