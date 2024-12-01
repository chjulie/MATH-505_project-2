import numpy as np
import scipy

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
        Z = np.linalg.lstsq(L, C.T)[0]
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
        Z = np.linalg.lstsq(L, Cperm.T)[0]
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


def create_sketch_matrix_gaussian(n: int, l: int) -> np.ndarray:
    return np.random.normal(loc=0.0, scale=1.0, size=[n, l])
