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
        Z = np.linalg.lstsq(L, C.T, rcond=None)[0]
        Z = Z.T
    except np.linalg.LinAlgError as err:
        # # Method1: Compute the SVD of B
        # U, S, _ = np.linalg.svd(B)  # For self-adjoint matrices, U = V
        # sqrt_S = np.sqrt(S)  # Compute square root of the singular values
        # # Construct the self-adjoint square root
        # sqrt_S_matrix = np.diag(sqrt_S)
        # L = U @ sqrt_S_matrix
        # # similarly as before
        # Z = np.linalg.lstsq(L, C.T)[0]
        # Z = Z.T

        # # Method2: Do LDL Factorization
        lu, d, perm = scipy.linalg.ldl(B)
        # Question for you: why is the following line not 100% correct?
        lu = lu @ np.sqrt(np.abs(d))
        # Does this factorization actually work?
        L = lu[perm, :]
        Cperm = C[:, perm]
        Z = np.linalg.lstsq(L, Cperm.T, rcond=None)[0]
        Z = Z.T

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
) -> Tuple[np.ndarray, np.ndarray]:

    # 1. Compute C = A @ Omega
    C = None
    if rank == 0:
        C = np.empty((n, l), dtype="float")
    C_local = A_local @ Omega_local
    rowmult = np.empty((n_local, l), dtype="float")
    comm_cols.Reduce(C_local, rowmult, op=MPI.SUM, root=0)
    if rank_cols == 0:
        comm_rows.Gather(rowmult, C, root=0)

    # 2.1 Compute B = Omega.T @ C
    B = None
    if rank == 0:
        B = np.empty((l, l), dtype="float")
    B_local = OmegaT_local @ A_local @ Omega_local
    comm.Reduce(B_local, B, op=MPI.SUM, root=0)

    return C, B

    # 2.2 Compute the Cholesky factorization of B: B = LL^T or eigen value decomposition

    # 3. Compute Z = C @ L.T with substitution

    # 4. Compute the QR factorization Z = QR

    # 5. Compute the truncated rank-k SVD of R: R = U Sigma V.T

    # 6. Compute U_hat = Q @ U

    # 7. Output factorization [A_nyst]_k = U_hat Sigma^2 U_hat.T

    return 0, 0


def create_sketch_matrix_gaussian_seq(n: int, l: int, seed: int = 10) -> np.ndarray:
    # TODO: check the numtiplication by 1/sqrt(n)
    np.random.seed(seed)
    return np.random.normal(loc=0.0, scale=1.0, size=[n, l])
    # return (1 / n) * np.random.normal(loc=0.0, scale=1.0, size=[n, l])


def create_sketch_matrix_SHRT_seq(n: int, l: int, seed=10) -> np.ndarray:
    np.random.seed(seed)
    # n x n => l x n
    d = np.array([1 if np.random.random() < 0.5 else -1 for _ in range(n)])
    D = np.diag(np.sqrt(n / l) * d)
    P = random.sample(range(n), l)
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

    Omega_T = Omega_T[P, :]
    return Omega_T.T


def is_power_of_two(n):
    if n <= 0:
        return False
    while n % 2 == 0:
        n //= 2
    return n == 1
