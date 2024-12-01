from mpi4py import MPI
import numpy as np

# 2D distribution for matrix−vector multiplication
# Initialize MPI (world)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n_blocks_row = int(np.sqrt(size))
n_blocks_col = int(np.sqrt(size))

if n_blocks_col**2 != int(size):
    if rank == 0:
        print("The number of processors must be a perfect square")
    exit(-1)

# # Define the matrix
# local_size = 2  # parameter hardcoded here
# n = int(np.sqrt(size)) * local_size
# matrix = np.arange(1, n * n + 1, 1, dtype="float")
# matrix = np.reshape(matrix, (n, n))
# arrs = np.split(matrix, n, axis=1)
# raveled = [np.ravel(arr) for arr in arrs]
# matrix_transpose = np.concatenate(raveled)

# nb_x_col = 3
# x = np.ones((n, nb_x_col))
# sol_mult_1 = np.empty((n, nb_x_col), dtype="float")
# sol_mult_2 = np.empty((nb_x_col, nb_x_col), dtype="float")
# sol_mult_3_1 = np.empty((n, nb_x_col), dtype="float")
# sol_mult_3_2 = np.empty((nb_x_col, nb_x_col), dtype="float")

# if rank == 0:
#     print(matrix)
#     print(x)

# comm_cols = comm.Split(color=rank / n_blocks_row, key=rank % n_blocks_row)
# comm_rows = comm.Split(color=rank % n_blocks_row, key=rank / n_blocks_row)


# in the context of this project, we will have the matrix A already built like this, and available at process 0
# now we need to broadcast it to the correst processors to make sure with can divide it by black after
# no need to broadcast it on every processor!

choice = "mult_3"  # mult_2, mult_3

# for the matrix
matrix = None
matrix_transpose = None
local_size = 2  # parameter hardcoded here
n = int(np.sqrt(size)) * local_size

# for x
x = None
nb_x_col = 2

# for sol
sol_mult_1 = None
sol_mult_2 = None
sol_mult_3_1 = None
sol_mult_3_2 = None

if rank == 0:  # only process 0 has the matrix initially
    matrix = np.arange(1, n * n + 1, 1, dtype="float")
    matrix = np.reshape(matrix, (n, n))
    arrs = np.split(matrix, n, axis=1)
    raveled = [np.ravel(arr) for arr in arrs]
    matrix_transpose = np.concatenate(raveled)

    x = np.ones((n, nb_x_col))
    sol_mult_1 = np.empty((n, nb_x_col), dtype="float")
    sol_mult_2 = np.empty((nb_x_col, nb_x_col), dtype="float")
    sol_mult_3_1 = np.empty((n, nb_x_col), dtype="float")
    sol_mult_3_2 = np.empty((nb_x_col, nb_x_col), dtype="float")

    print(matrix)
    print(x)


comm_cols = comm.Split(color=rank / n_blocks_row, key=rank % n_blocks_row)
comm_rows = comm.Split(color=rank % n_blocks_row, key=rank / n_blocks_row)

matrix = comm_rows.bcast(matrix, root=0)
matrix_transpose = comm_rows.bcast(matrix_transpose, root=0)
x = comm_rows.bcast(x, root=0)
x = comm_cols.bcast(x, root=0)  # for mult_2 and mult_3, no need for mult_1

# print(
#     "Original rank: ",
#     rank,
#     "matrix: ",
#     matrix,
#     "x: ",
#     x,
# )

# print(
#     "Original rank: ",
#     rank,
#     "color col",
#     int(rank / n_blocks_row),
#     "color row ",
#     int(rank % n_blocks_row),
# )

# Get ranks of subcommunicator
rank_cols = comm_cols.Get_rank()
rank_rows = comm_rows.Get_rank()

# DISTRIBUTE THE MATRIX
# Select columns
submatrix = np.empty((local_size, n), dtype="float")

# Then we scatter the columns and put them in the right order
receiveMat = np.empty((local_size * n), dtype="float")
comm_cols.Scatterv(matrix_transpose, receiveMat, root=0)
subArrs = np.split(receiveMat, local_size)
raveled = [np.ravel(arr, order="F") for arr in subArrs]
submatrix = np.ravel(raveled, order="F")

# Then we scatter the rows
blockMatrix = np.empty((local_size, local_size), dtype="float")
comm_rows.Scatterv(submatrix, blockMatrix, root=0)

print(
    "Original rank: ",
    rank,
    " rank in splitrows: ",
    rank_rows,
    " rank in splitcols: ",
    rank_cols,
    "submatrix after scattering columns: ",
    submatrix,
    "block matrix: ",
    blockMatrix,
    "\n\n",
)


if choice == "mult_1":

    # Parallel Matrix Multiplication A @ X
    # distribute X using columns
    x_block = np.empty((local_size, nb_x_col), dtype="float")
    comm_cols.Scatterv(x, x_block, root=0)
    # Multiply in place each block matrix with each X block
    local_mult = blockMatrix @ x_block
    # Now sum those local multiplications along rows
    rowmult = np.empty((local_size, nb_x_col), dtype="float")
    comm_cols.Reduce(local_mult, rowmult, op=MPI.SUM, root=0)

    if rank_cols == 0:
        comm_rows.Gather(rowmult, sol_mult_1, root=0)

    # Print in the root process
    if rank == 0:
        print(
            "Solution mult 1 with MPI: ",
            sol_mult_1,
            "Solution mult 1 with Python: ",
            matrix @ x,
        )

elif choice == "mult_2":

    # Parallel Matrix Multiplication X^T A X
    # distribute X using columns and X^T using rows
    x_block = np.empty((local_size, nb_x_col), dtype="float")
    comm_cols.Scatterv(x, x_block, root=0)

    x_T_block = np.empty((local_size, nb_x_col), dtype="float")
    comm_rows.Scatterv(x, x_T_block, root=0)
    x_T_block = x_T_block.T

    local_mult = x_T_block @ blockMatrix @ x_block

    comm.Reduce(local_mult, sol_mult_2, op=MPI.SUM, root=0)

    # Print in the root process
    if rank == 0:
        print(
            "Solution mult 2 with MPI: ",
            sol_mult_2,
            "Solution mult 2 with Python: ",
            x.T @ matrix @ x,
        )


elif choice == "mult_3":

    # Both parallel multiplication at the same time now
    x_block = np.empty((local_size, nb_x_col), dtype="float")
    comm_cols.Scatterv(x, x_block, root=0)

    x_T_block = np.empty((local_size, nb_x_col), dtype="float")
    comm_rows.Scatterv(x, x_T_block, root=0)
    x_T_block = x_T_block.T

    local_mult_1 = blockMatrix @ x_block
    local_mult_2 = x_T_block @ blockMatrix @ x_block

    # sol mult 1
    rowmult = np.empty((local_size, nb_x_col), dtype="float")
    comm_cols.Reduce(local_mult_1, rowmult, op=MPI.SUM, root=0)
    if rank_cols == 0:
        comm_rows.Gather(rowmult, sol_mult_3_1, root=0)

    # sol mult 2
    comm.Reduce(local_mult_2, sol_mult_3_2, op=MPI.SUM, root=0)

    # Print in the root process
    if rank == 0:
        print(
            "Solution mult 1 with MPI: ",
            sol_mult_3_1,
            "Solution mult 1 with Python: ",
            matrix @ x,
            "Solution mult 2 with MPI: ",
            sol_mult_3_2,
            "Solution mult 2 with Python: ",
            x.T @ matrix @ x,
        )

else:
    print("incorrect choice")
