import numpy as np
import scipy
import random

# import torch
# from hadamard_transform import hadamard_transform
from scipy.linalg import hadamard
from typing import Tuple
from mpi4py import MPI


def nuclear_error(A, A_nyst, k):
    # TODO compute error of rank-k truncation of the Nystroem approx. using the nuclear norm

    return

def rand_nystrom_seq(
    A: np.ndarray, Omega: np.ndarray, k: int, return_extra: bool = True
) -> np.ndarray:

    C = A @ Omega
    B = Omega.T @ C

    try:
        # Try Cholesky
        L = np.linalg.cholesky(B)
        Z = np.linalg.lstsq(L, C.T, rcond=-1)[0]
        Z = Z.T
    except np.linalg.LinAlgError as err:
        # # Method 1: Compute the SVD of B
        # U, S, _ = np.linalg.svd(B)  # For self-adjoint matrices, U = V
        # sqrt_S = np.sqrt(S)  # Compute square root of the singular values
        # # Construct the self-adjoint square root
        # sqrt_S_matrix = np.diag(sqrt_S)
        # L = U @ sqrt_S_matrix
        # # similarly as before
        # Z = np.linalg.lstsq(L, C.T)[0]
        # Z = Z.T

        # Method 2: Do LDL Factorization
        lu, d, perm = scipy.linalg.ldl(B)
        # Question for you: why is the following line not 100% correct?
        lu = lu @ np.sqrt(np.abs(d))
        # Does this factorization actually work?
        L = lu[perm, :]
        Cperm = C[:, perm]
        Z = np.linalg.lstsq(L, Cperm.T, rcond=-1)[0]
        Z = Z.T

        # Method 3: Use eigen value decomposition:
        # eigenvalues, eigenvectors = np.linalg.eig(B)
        # sqrt_eigenvalues = np.sqrt(np.abs(eigenvalues))  # Ensure numerical stability
        # L = eigenvectors @ np.diag(sqrt_eigenvalues)
        # Z = np.linalg.lstsq(L, C.T, rcond=-1)[0]
        # Z = Z.T

    Q, R = np.linalg.qr(Z)
    U_tilde, S, V = np.linalg.svd(R)
    Sigma = np.diag(S)
    U_hat = Q @ U_tilde

    # perform rank k truncating
    U_hat_k = U_hat[:, :k]
    Sigma_k = Sigma[:k, :k]

    if return_extra:
        S_B = np.linalg.cond(B)
        rank_A = np.linalg.matrix_rank(A)
        return U_hat_k, Sigma_k @ Sigma_k, U_hat_k.T, S_B, rank_A
    else:
        return U_hat_k, Sigma_k @ Sigma_k, U_hat_k.T


def rand_nystrom_parallel(
    A_local: np.ndarray,
    Omega_local: np.ndarray,
    OmegaT_local: np.ndarray,
    k: int,
    n: int,
    n_local: int,
    l: int,
    comm,
    comm_cols,
    comm_rows,
    rank,
    rank_cols,
    rank_rows,
    size_rows,
) -> Tuple[np.ndarray, np.ndarray]:

    # 1. Compute C = A @ Omega
    C = None
    if rank == 0:
        C = np.empty((n, l), dtype="float")
    C_local_subpart = A_local @ Omega_local
    C_local = np.empty(
        (n_local, l), dtype="float"
    )  # TODO: is it the best way to do it??
    comm_cols.Reduce(C_local_subpart, C_local, op=MPI.SUM, root=0)
    if (
        rank_cols == 0
    ):  # TODO: this might not even be necessary, because technically no need to assemble C
        comm_rows.Gather(C_local, C, root=0)

    # 2.1 Compute B = Omega.T @ C
    B = None
    if rank == 0:
        B = np.empty((l, l), dtype="float")
    B_local = OmegaT_local @ A_local @ Omega_local
    comm.Reduce(B_local, B, op=MPI.SUM, root=0)

    L = None
    permute = False
    perm = None
    # 2.2 Compute the Cholesky factorization of B: B = LL^T or eigen value decomposition
    if rank == 0:  # compute only at the root
        try:  # Try Cholesky
            L = np.linalg.cholesky(B)
        except np.linalg.LinAlgError as err:
            # Do LDL Factorization (TODO: might need to do change)
            lu, d, perm = scipy.linalg.ldl(B)
            # Question for you: why is the following line not 100% correct?
            lu = lu @ np.sqrt(np.abs(d))
            # Does this factorization actually work?
            L = lu[perm, :]
            permute = True

    L = comm_rows.bcast(L, root=0)  # broadcast through rows
    perm = comm_rows.bcast(perm, root=0)
    if permute == True and rank_cols == 0:
        C_local = C_local[:, perm]

    # 3. Compute Z = C @ L.T with substitution
    # this is only computed in processes of the first column (with rank_cols = 0)
    Z_local = None
    if rank_cols == 0:
        Z_local = np.linalg.lstsq(L, C_local.T, rcond=-1)[0]
        Z_local = Z_local.T

    # 4. Compute the QR factorization Z = QR
    R = None
    Q_local = None
    if rank_cols == 0:
        Q_local, R = TSQR(Z_local, l, comm_rows, rank_rows, size_rows)

    # 5. Compute the truncated rank-k SVD of R: R = U Sigma V.T
    U_tilde = None
    S = None
    Sigma_2 = None
    if rank == 0:
        U_tilde, S, V = np.linalg.svd(R)
        Sigma = np.diag(S)
        # truncate to get rank k
        U_tilde = U_tilde[:, :k]
        Sigma = Sigma[:k, :k]
        Sigma_2 = Sigma @ Sigma

    U_tilde = comm_rows.bcast(U_tilde, root=0)  # broadcast through rows

    # 6. Compute U_hat = Q @ U
    U_hat_local = np.empty((A_local.shape[0], k), dtype="float")
    if rank_cols == 0:
        U_hat_local = Q_local @ U_tilde

    # 7. Output factorization [A_nyst]_k = U_hat Sigma^2 U_hat.T
    return U_hat_local, Sigma_2


def create_sketch_matrix_gaussian_seq(n: int, l: int, seed: int = 0) -> np.ndarray:
    np.random.seed(seed)
    return np.random.normal(loc=0.0, scale=1.0, size=[n, l])


def create_sketch_matrix_gaussian_parallel(
    n_local: int, l: int, seed: int
) -> np.ndarray:
    np.random.seed(seed)
    return np.random.normal(loc=0.0, scale=1.0, size=[n_local, l])


def create_sketch_matrix_SHRT_seq(n: int, l: int, seed: int = 0) -> np.ndarray:
    np.random.seed(seed)
    # n x n => l x n
    d = np.array([1 if np.random.random() < 0.5 else -1 for _ in range(n)])
    D = np.diag(np.sqrt(n / l) * d)
    # random.seed(seed)
    # R = random.sample(range(n), l)
    R = np.random.choice(range(n), size=l)
    Omega_T = D

    # using scipy
    H = hadamard(Omega_T.shape[0]) / np.sqrt(Omega_T.shape[0])  # normalize
    Omega_T = np.array([H.T @ Omega_T[:, i] for i in range(n)])
    Omega_T = Omega_T.T

    # using torch
    # Omega_T = np.array(
    #     [hadamard_transform(torch.from_numpy(Omega_T[:, i])).numpy() for i in range(n)]
    # )
    # Omega_T = Omega_T.T
    # print(np.allclose(Omega_T_1, Omega_T, rtol=10e-6))

    Omega_T = Omega_T[R, :]
    return Omega_T.T


def create_sketch_matrix_SHRT_parallel(
    n_local: int, l: int, seed_local: int, seed_global: int
) -> np.ndarray:
    # use local seed to compute diagobal matrices
    np.random.seed(seed_local)
    dr = np.array([1 if np.random.random() < 0.5 else -1 for _ in range(n_local)])
    dl = np.array([1 if np.random.random() < 0.5 else -1 for _ in range(l)])
    DR = np.diag(np.sqrt(n_local / l) * dr)
    DL = np.diag(dl)

    # use global seed to select rows
    random.seed(seed_global)
    R = random.sample(range(n_local), l)
    # R = np.random.choice(range(n_local), size=l)

    Omega_T = DR
    H = hadamard(Omega_T.shape[0]) / np.sqrt(Omega_T.shape[0])  # normalize
    Omega_T = np.array([H.T @ Omega_T[:, i] for i in range(n_local)])
    Omega_T = Omega_T.T
    Omega_T = Omega_T[R, :]

    Omega_T = DL @ Omega_T

    return Omega_T.T


def is_power_of_two(n):
    if n <= 0:
        return False
    while n % 2 == 0:
        n //= 2
    return n == 1


def TSQR(W_local, n, comm, rank, size):
    R = None

    # At first step, compute local Householder QR
    Q_local, R_local = np.linalg.qr(W_local)  # sequential QR with numpy

    # Store the Q factors generated at each depth level
    Q_factors = [Q_local]
    depth = int(np.log2(size))

    for k in range(depth):
        I = int(rank)

        # processes that need to exit the loop
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
            )  # this tag makes sure it is the same for both partners
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
