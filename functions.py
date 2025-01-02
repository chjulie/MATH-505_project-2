import numpy as np
import scipy
import random
import math
import time

from typing import Tuple
from mpi4py import MPI


def nuclear_error(A: np.ndarray, U: np.ndarray, Sigma_2: np.ndarray) -> float:
    err_nuclear = np.linalg.norm(U @ Sigma_2 @ U.T - A, ord="nuc") / np.linalg.norm(
        A, ord="nuc"
    )
    return err_nuclear


def rand_nystrom_sequential(
    A: np.ndarray,
    seed: int,
    n: int,
    sketching: str,
    k: int,
    l: int,
    return_extra: bool = False,
    return_runtimes: bool = False,
    print_computation_times: bool = True,
) -> np.ndarray:

    t1 = time.time()
    np.random.seed(seed)
    random.seed(seed)

    if sketching == "SHRT":
        # C = (Ω × A).T
        d = np.array([1 if np.random.random() < 0.5 else -1 for _ in range(n)])
        C = np.multiply(np.sqrt(n / l) * d[:, np.newaxis], A)
        C = np.array([FWHT(C[:, i]) for i in range(n)]).T
        R = random.sample(range(n), l)
        C = C[R, :].T

        # B = Ω × C
        B = np.multiply(np.sqrt(n / l) * d[:, np.newaxis], C)
        B = np.array([FWHT(B[:, i]) for i in range(l)]).T
        B = B[R, :]

    elif sketching == "gaussian":

        Omega = np.random.normal(loc=0.0, scale=1.0, size=[l, n])
        # C = (Ω × A).T
        C = (Omega @ A).T
        # B = Ω × C
        B = Omega @ C

    else:
        raise (NotImplementedError)

    t2 = time.time()

    try:
        # Try Cholesky
        L = np.linalg.cholesky(B)
    except np.linalg.LinAlgError as err:
        # Method 1: Do LDL Factorization
        lu, d, perm = scipy.linalg.ldl(B)
        lu = lu @ np.sqrt(np.abs(d))  # for stability
        L = lu[perm, :]
        C = C[:, perm]

        # Method 2: Use eigen value decomposition:
        # eigenvalues, eigenvectors = np.linalg.eig(B)
        # sqrt_eigenvalues = np.sqrt(np.abs(eigenvalues))  # Ensure numerical stability
        # L = eigenvectors @ np.diag(sqrt_eigenvalues)
    t3 = time.time()

    Z = np.linalg.lstsq(L, C.T, rcond=-1)[0]
    Z = Z.T
    t4 = time.time()

    Q, R = np.linalg.qr(Z)
    t5 = time.time()

    U_tilde, S, V = np.linalg.svd(R)
    Sigma = np.diag(S)
    U_hat = Q @ U_tilde
    # Perform rank k truncating
    U_hat_k = U_hat[:, :k]
    Sigma_k = Sigma[:k, :k]
    t6 = time.time()

    # PRINT OUT COMPUTATION TIMES
    runtimes = [
        t2-t1, 
        t3-t2, 
        t4-t3, 
        t5-t4, 
        t6-t5
    ]

    if print_computation_times:
        print("\n ** COMPUTATION TIMES ** \n")
        print(f" - Apply Ω: B = Ω (Ω A).T: {runtimes[0]:.4f} s.")
        print(f" - Cholesky decomposition: B = L L.T: {runtimes[1]:.4f} s.")
        print(f" - Z with substitution: Z = C @ L.-T: {runtimes[2]:.4f} s.")
        print(f" - QR factorization: {runtimes[3]:.4f} s.")
        print(f" - Truncated rank-r SVD: {runtimes[4]:.4f} s.\n")

    if return_extra:
        S_B = np.linalg.cond(B)
        rank_A = np.linalg.matrix_rank(A)
        if return_runtimes:
            return U_hat_k, Sigma_k @ Sigma_k, S_B, rank_A, runtimes
        else:
            return U_hat_k, Sigma_k @ Sigma_k, S_B, rank_A
    else:
        if return_runtimes:
            return U_hat_k, Sigma_k @ Sigma_k, runtimes
        else:
            return U_hat_k, Sigma_k @ Sigma_k


def SFWHT(a: np.ndarray) -> np.ndarray:
    """Fast Walsh–Hadamard Transform of vector a
    Slowest version (but more memory efficient).

    Inspired from the Wikipedia implementation: https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform
    """
    # assert math.log2(len(a)).is_integer(), "length of a is a power of 2"
    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return a / math.sqrt(len(a))


def FWHT(x: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform
    Based on mex function written by Chengbo Li@Rice Uni for his TVAL3 algorithm.
    His code is according to the K.G. Beauchamp's book -- Applications of Walsh and Related Functions.

    credits: https://github.com/dingluo/fwht
    """
    x = x.squeeze()
    N = x.size
    G = int(N / 2)  # Number of Groups
    M = 2  # Number of Members in Each Group

    # First stage
    y = np.zeros((int(N / 2), 2))
    y[:, 0] = x[0::2] + x[1::2]
    y[:, 1] = x[0::2] - x[1::2]
    x = y.copy()

    # Second and further stage
    for nStage in range(2, int(math.log(N, 2)) + 1):
        y = np.zeros((int(G / 2), int(M * 2)))
        G = int(G)
        M = int(M)
        y[0 : int(G / 2), 0 : int(M * 2) : 4] = x[0:G:2, 0:M:2] + x[1:G:2, 0:M:2]
        y[0 : int(G / 2), 1 : int(M * 2) : 4] = x[0:G:2, 1:M:2] + x[1:G:2, 1:M:2]
        y[0 : int(G / 2), 2 : int(M * 2) : 4] = x[0:G:2, 0:M:2] - x[1:G:2, 0:M:2]
        y[0 : int(G / 2), 3 : int(M * 2) : 4] = x[0:G:2, 1:M:2] - x[1:G:2, 1:M:2]
        x = y.copy()
        G = G / 2
        M = M * 2
    x = y[0, :]
    x = x.reshape((x.size, 1)).squeeze(-1)
    return x / math.sqrt(N)


def rand_nystrom_parallel(
    A_local: np.ndarray,
    seed_global: int,
    k: int,
    n: int,
    n_local: int,
    l: int,
    sketching: str,
    comm,
    comm_cols,
    comm_rows,
    rank: int,
    rank_cols: int,
    rank_rows: int,
    size_cols: int,
    print_computation_times: bool = True,
    return_runtimes: bool = False
) -> Tuple[np.ndarray, np.ndarray]:

    t1 = time.time()

    if sketching == "SHRT":
        # Share the seed amongst rows (distribute Ω over the columns)
        seed_local = rank_rows
        np.random.seed(seed_local)

        C = None
        if rank_rows == 0:
            C = np.empty((l, n_local), dtype="float")
        dr = np.array([1 if np.random.random() < 0.5 else -1 for _ in range(n_local)])
        dl = np.array([1 if np.random.random() < 0.5 else -1 for _ in range(l)])

        # 1. Compute C = Ω × A
        # A = DR @ A
        C_local = None
        C_local = np.multiply(np.sqrt(n_local / l) * dr[:, np.newaxis], A_local)
        # A = H @ A => apply the transform instead of computing the matrix explicitly
        # !! list comprehension construction swaps axis 0 and 1!!
        C_local = np.array([FWHT(C_local[:, i]) for i in range(n_local)]).T
        # A = R @ A
        # Use global seed to select rows
        random.seed(seed_global)
        R = random.sample(range(n_local), l)
        C_local = C_local[R, :]
        # Compute C = DL R H DR A
        C_local = np.multiply(dl[:, np.newaxis], C_local)
        # Sum-reduce by rows
        comm_rows.Reduce(C_local, C, op=MPI.SUM, root=0)

        # 2.1 Compute B = Ω × C.T
        B = None
        if rank == 0:
            B = np.empty((l, l), dtype="float")

        if rank_rows == 0:
            # Apply Ω matrix
            # Share the seed amongst columns (distribute Ω over the rows)
            seed_local = rank_cols
            np.random.seed(seed_local)
            dr = np.array(
                [1 if np.random.random() < 0.5 else -1 for _ in range(n_local)]
            )
            dl = np.array([1 if np.random.random() < 0.5 else -1 for _ in range(l)])

            B_local = np.multiply(np.sqrt(n_local / l) * dr[:, np.newaxis], C.T)
            # !! list comprehension construction swaps axis 0 and 1!!
            B_local = np.array(
                [FWHT(B_local[:, i]) for i in range(l)]
            ).T  # Only l columns instead of n_local now
            B_local = B_local[R, :]
            B_local = np.multiply(dl[:, np.newaxis], B_local)

            comm_cols.Reduce(B_local, B, op=MPI.SUM, root=0)

            # if rank == 0:
            #     print(" B.shape: ", B.shape)
            #     print(" C.shape: ", C.shape)
            # print(f" * Rank {rank}, rank_cols: {rank_cols}, rank_rows: {rank_rows}: B: {B}\n")

    elif sketching == "gaussian":

        np.random.seed(rank_rows)

        C = None
        if rank_rows == 0:
            C = np.empty((l, n_local), dtype="float")

        # 1. Compute C = Ω × A
        # Generate gaussian matrix
        C_local = None
        C_local = np.random.normal(loc=0.0, scale=1.0, size=[l, n_local]) @ A_local

        comm_rows.Reduce(C_local, C, op=MPI.SUM, root=0)
        # print(f" * Rank {rank}, rank_cols: {rank_cols}, rank_rows: {rank_rows}: C: {C}\n")

        # 2.1 Compute B = Ω × C.T
        B = None
        if rank_cols == 0:
            B = np.empty((l, l), dtype="float")

        if rank_rows == 0:
            # Apply Ω matrix
            # Share the seed amongst columns (distribute Ω over the rows)
            np.random.seed(rank_cols)

            B_local = np.random.normal(loc=0.0, scale=1.0, size=[l, n_local]) @ C.T
            comm_cols.Reduce(B_local, B, op=MPI.SUM, root=0)
            # print(f" * Rank {rank}, rank_cols: {rank_cols}, rank_rows: {rank_rows}: B: {B}\n")

    else:
        raise (NotImplementedError)

    t2 = time.time()

    L = None
    permute = False
    perm = None
    # 2.2 Compute the Cholesky factorization of B: B = LL^T or LDL decomposition
    if rank == 0:  # compute only at the root
        try:  # Try Cholesky
            L = np.linalg.cholesky(B)
            # print(" > Cholesky succeeded!")
        except np.linalg.LinAlgError as err:
            # Do LDL Factorization
            lu, d, perm = scipy.linalg.ldl(B)
            lu = lu @ np.sqrt(np.abs(d))
            L = lu[perm, :]
            permute = True
            # print(" > LDL factorization succeeded!")

    L = comm_cols.bcast(L, root=0)  # Broadcast through columns
    perm = comm_cols.bcast(perm, root=0)
    if permute == True and rank_rows == 0:
        C = C[perm, :]

    t3 = time.time()

    # 3. Compute Z = C @ L.-T with substitution
    # This is only computed in processes of the first row (with rank_rows = 0)
    Z_local = None
    if rank_rows == 0:
        # C and not C.T since our C are stored transposed
        Z_local = np.linalg.lstsq(L, C, rcond=-1)[0]
        Z_local = Z_local.T

    t4 = time.time()

    # 4. Compute the QR factorization Z = QR
    R = None
    Q_local = None
    if rank_rows == 0:
        Q_local, R = TSQR(Z_local, l, comm_cols, rank_cols, size_cols)

    t5 = time.time()

    # 5. Compute the truncated rank-k SVD of R: R = U Sigma V.T
    U_tilde = None
    S = None
    Sigma_2 = None
    if rank == 0:
        U_tilde, S, V = np.linalg.svd(R)

        # truncate to get rank k
        S_2 = S[:k] * S[:k]
        Sigma_2 = np.diag(S_2)
        U_tilde = U_tilde[:, :k]

    U_tilde = comm_cols.bcast(U_tilde, root=0)  # broadcast through cols

    # 6. Compute U_hat = Q @ U
    U_hat_local = None
    if rank_rows == 0:
        U_hat_local = Q_local @ U_tilde
        # print(f" * Rank {rank}, rank_cols: {rank_cols}, rank_rows: {rank_rows}: U_local: {U_hat_local}\n")

    t6 = time.time()

    # PRINT OUT COMPUTATION TIMES
    runtimes = [
        t2-t1, 
        t3-t2, 
        t4-t3, 
        t5-t4, 
        t6-t5
    ]

    if rank == 0 and print_computation_times:
        print("\n ** COMPUTATION TIMES ** \n")
        print(f" - Apply Ω: B = Ω (Ω A).T: {runtimes[0]:.4f} s.")
        print(f" - Cholesky decomposition: B = L L.T: {runtimes[1]:.4f} s.")
        print(f" - Z with substitution: Z = C @ L.-T: {runtimes[2]:.4f} s.")
        print(f" - QR factorization: {runtimes[3]:.4f} s.")
        print(f" - Truncated rank-r SVD: {runtimes[4]:.4f} s.\n")

    # 7. Output factorization [A_nyst]_k = U_hat Sigma^2 U_hat.T
    if return_runtimes:
        return U_hat_local, Sigma_2, runtimes
    else:
        return U_hat_local, Sigma_2


def is_power_of_two(n: int) -> bool:
    if n <= 0:
        return False
    while n % 2 == 0:
        n //= 2
    return n == 1


def TSQR(
    W_local: np.ndarray, n: int, comm, rank: int, size: int
) -> tuple[np.ndarray, np.ndarray]:
    R = None

    # At first step, compute local Householder QR
    Q_local, R_local = np.linalg.qr(W_local)  # sequential QR with numpy

    # Store the Q factors generated at each depth level
    Q_factors = [Q_local]
    depth = int(np.log2(size))

    for k in range(depth):
        I = int(rank)

        # Processes that need to exit the loop
        # are the processes that has a neighbor I - 2**k in the previous loop
        # also do not remove any process at the first iteration
        if (k != 0) and ((I % (2 ** (k))) >= 2 ** (k - 1)):
            break

        if (I % (2 ** (k + 1))) < 2**k:
            J = I + 2**k
        else:
            J = I - 2**k

        if I > J:
            comm.send(
                R_local, dest=J, tag=I + J
            )  # This tag makes sure it is the same for both partners
        else:
            other_R_local = comm.recv(source=J, tag=I + J)
            new_R = np.vstack((R_local, other_R_local))
            Q_local, R_local = np.linalg.qr(new_R)
            Q_factors.insert(0, Q_local)

    comm.Barrier()  # make sure all have finished

    nb_Q_factors_local = len(Q_factors)

    # Now need to compute Q
    # Get Q in reverse order, starting from root to the leaves
    i_local = 0
    nb_Q_factors_local = len(Q_factors)
    if rank == 0:
        R = R_local  # R matrix was computed already, stored in process 0
        Q_local = Q_factors[i_local]  # Q is intialized to last Q_local
        i_local += 1

    for k in range(depth - 1, -1, -1):
        # processes sending
        if nb_Q_factors_local > k + 1:
            I = int(rank)
            J = int(I + 2**k)
            rhs = Q_local[:n, :]
            to_send = Q_local[n:, :]
            comm.send(to_send, dest=J)

        # processes receiving
        if nb_Q_factors_local == (k + 1):
            I = int(rank)
            J = int(I - 2**k)
            rhs = np.zeros((n, n), dtype="d")
            rhs = comm.recv(source=J)

        # processes doing multiplications
        if nb_Q_factors_local >= k + 1:
            Q_local = Q_factors[i_local] @ rhs
            i_local += 1
    return Q_local, R
