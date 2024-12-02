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

    comm_cols = comm.Split(color=rank / n_blocks_row, key=rank % n_blocks_row)
    comm_rows = comm.Split(color=rank % n_blocks_row, key=rank / n_blocks_row)
    # Get ranks of subcommunicator
    rank_cols = comm_cols.Get_rank()
    rank_rows = comm_rows.Get_rank()

    # GENERATE THE MATRIX A AND OMEGA
    # parameters
    n = 1024
    n_local = int(n / n_blocks_row)
    l = 50
    Rs = [5, 10, 20]
    ks = [50, 50, 50]  # k < l!!!
    ps = [0.5, 1, 2]
    qs = [0.1, 0.25, 1.0]
    sketching = "gaussian"  # "gaussian", "SHRT"
    # for 1 matrix testing
    R = Rs[0]
    p = ps[0]
    k = ks[0]

    # test: TODO: remove
    # # parameters
    # n = 8
    # n_local = int(n / n_blocks_row)
    # l = 4
    # Rs = [2, 3, 4]
    # ks = [2, 3, 4]  # k < l!!!
    # ps = [0.5, 1, 2]
    # qs = [0.1, 0.25, 1.0]
    # sketching = "gaussian"  # "gaussian", "SHRT"

    if n_blocks_col * n_local != n:  # Check n is divisible by n_blocks_row
        print("n should be divisible by sqrt(P) where P is the number of processors")
    if not is_power_of_two(n):  # Check n is a power of 2
        print("n should be a power of 2")

    # Initialization
    A = None
    AT = None
    C = None
    B = None
    Omega = None

    # Generate A
    if rank == 0:
        A = pol_decay(n, R, p)
        arrs = np.split(A, n, axis=1)
        raveled = [np.ravel(arr) for arr in arrs]
        AT = np.concatenate(raveled)
        print("Shape of A: ", A.shape)
    AT = comm_rows.bcast(AT, root=0)
    # TODO: is it better to generate the matrix at every processor? or generate at root and broadcast

    # Generate Omega
    if rank == 0:
        # test if needed TODO: remove
        # Omega = np.arange(1, n * l + 1, 1, dtype="float")
        # Omega = np.reshape(Omega, (n, l))
        if sketching == "gaussian":
            Omega = create_sketch_matrix_gaussian_seq(n, l)
        elif sketching == "SHRT":
            Omega = create_sketch_matrix_SHRT_seq(n, l)
        print("Shape of Omega: ", Omega.shape)
    Omega = comm_rows.bcast(Omega, root=0)
    Omega = comm_cols.bcast(Omega, root=0)
    # TODO: generate omega IN PARALLEL! they are currently generated at root and then broadcasted

    # DISTRIBUTE THE MATRIX A TO GET A_local
    # Select columns, scatter them and put them in the right order
    submatrix = np.empty((n_local, n), dtype="float")
    receiveMat = np.empty((n_local * n), dtype="float")
    comm_cols.Scatterv(AT, receiveMat, root=0)
    subArrs = np.split(receiveMat, n_local)
    raveled = [np.ravel(arr, order="F") for arr in subArrs]
    submatrix = np.ravel(raveled, order="F")
    # Scatter the rows
    A_local = np.empty((n_local, n_local), dtype="float")
    comm_rows.Scatterv(submatrix, A_local, root=0)

    # DISTRIBUTE OMEGA TO GET Omega_local and OmegaT_local
    Omega_local = np.empty((n_local, l), dtype="float")
    comm_cols.Scatterv(Omega, Omega_local, root=0)

    OmegaT_local = np.empty((n_local, l), dtype="float")
    comm_rows.Scatterv(Omega, OmegaT_local, root=0)
    OmegaT_local = OmegaT_local.T

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

    C, B = rand_nystrom_parallel(
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
    )

    # Print in the root process
    if rank == 0:
        if np.allclose(A @ Omega, C, atol=1e-6):  # small tolerance level
            print("C: Success!")
        else:
            print("C: Error!")
        if np.allclose(Omega.T @ A @ Omega, B, atol=1e-6):  # small tolerance level
            print("B: Success!")
        else:
            print("B: Error!")

        # print(
        #     "Solution for C with MPI: ",
        #     C,
        #     "Solution for C with Python: ",
        #     A @ Omega,
        #     "Solution for B with MPI: ",
        #     B,
        #     "Solution for B with Python: ",
        #     Omega.T @ A @ Omega,
        # )

    # U, Sigma = rand_nystrom_parallel(A_local, Omega_local, OmegaT_local, ks[0])
