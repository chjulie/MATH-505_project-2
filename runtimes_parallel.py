import numpy as np
from mpi4py import MPI
import json

from data_helpers import pol_decay, exp_decay
from functions import (
    is_power_of_two,
    rand_nystrom_parallel,
)

if __name__ == "__main__":

    # INITIALIZE MPI WORLD
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

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

    # INITIAIZE DATA STORAGE
    json_file = "results/runtimes_" + str(size) + ".json"
    data = []

    # LOOP OVER MATRIX SIZES
    ns = [8192]  # [1024, 2048, 4096, 8192]

    for n in ns:
        if rank == 0:
            print(f" > n = {n}")

        # INITIALIZATION
        A = None
        AT = None

        A_choice = "mnist"
        n_local = int(n / n_blocks_row)
        seed_global = 42
        l = 128
        k = 100  # k <=l !! + does not influence runtime

        # Check the size of A
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

        # GENERATE THE MATRIX A
        if A_choice == "exp_decay" or A_choice == "pol_decay":
            # Generate at root and then broadcast
            if rank == 0:
                R = 10
                if A_choice == "exp_decay":
                    q = 0.1
                    A = exp_decay(n, R, q)
                else:
                    p = 0.5
                    A = pol_decay(n, R, p)
                AT = A  # Matrix is SPD
                print("Shape of A: ", A.shape)
            AT = comm_rows.bcast(AT, root=0)

        elif A_choice == "mnist":
            if rank == 0:
                A = np.load("data/mnist_" + str(n) + ".npy")
                AT = A  # Matrix is SPD
                print("Shape of A: ", A.shape)
            AT = comm_rows.bcast(AT, root=0)
        else:
            raise (NotImplementedError)

        # 1. Distribute A over processors
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

        it = 5
        average_runtimes_gaussian = np.array(np.zeros(5))
        average_runtimes_SHRT = np.array(np.zeros(5))

        # Gaussian sketching matrix
        if rank == 0:
            print("  > gaussian sketching")
        for i in range(it):

            _, _, runtimes = rand_nystrom_parallel(
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
                return_runtimes=True,
                print_computation_times=False,
            )

            average_runtimes_gaussian += runtimes
            if rank == 0:
                print(f"   > it {i+1} done")

        average_runtimes_gaussian = average_runtimes_gaussian / it

        # SHRT sketching matrix
        if rank == 0:
            print("  > SHRT sketching matrix")
        for i in range(it):

            _, _, runtimes = rand_nystrom_parallel(
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
                return_runtimes=True,
                print_computation_times=False,
            )

            average_runtimes_SHRT += runtimes
            if rank == 0:
                print(f"   > it {i+1} done")

        average_runtimes_SHRT = average_runtimes_SHRT / it

        run_details = {
            "matrix_size": n,
            "n_proc": size,
            "runtimes_gaussian": average_runtimes_gaussian.tolist(),
            "runtimes_SHRT": average_runtimes_SHRT.tolist(),
        }
        data.append(run_details)

    if rank == 0:
        with open(json_file, "w") as file:
            json.dump(data, file, indent=4)

        print(" > Program finished successfully!")
