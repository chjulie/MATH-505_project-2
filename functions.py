import numpy as np
import scipy
import random
import torch
from hadamard_transform import hadamard_transform

# from mpi4py import MPI


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


def rand_nystrom_parallel(A: np.ndarray, Omega: np.ndarray, k: int) -> np.ndarray:
    # 1. Compute C = A @ Omega

    # 2.1 Compute B = Omega.T @ C

    # 2.2 Compute the Cholesky factorization of B: B = LL^T or eigen value decomposition

    # 3. Compute Z = C @ L.T with substitution

    # 4. Compute the QR factorization Z = QR

    # 5. Compute the truncated rank-k SVD of R: R = U Sigma V.T

    # 6. Compute U_hat = Q @ U

    # 7. Output factorization [A_nyst]_k = U_hat Sigma^2 U_hat.T
    pass


def create_sketch_matrix_gaussian(n: int, l: int) -> np.ndarray:
    # TODO: check the numtiplication by 1/sqrt(n)
    # return (1 / np.sqrt(n)) * np.random.normal(loc=0.0, scale=1.0, size=[n, l])
    return np.random.normal(loc=0.0, scale=1.0, size=[n, l])


def create_sketch_matrix_SHRT(n: int, l: int) -> np.ndarray:
    # n x n => l x n
    d = np.array([1 if random.random() < 0.5 else -1 for _ in range(n)])
    D = np.diag(np.sqrt(n / l) * d)
    P = random.sample(range(n), l)
    Omega_T = D
    Omega_T = np.array(
        [hadamard_transform(torch.from_numpy(Omega_T[:, i])).numpy() for i in range(n)]
    )
    Omega_T = Omega_T.T
    Omega_T = Omega_T[P, :]
    return Omega_T.T
