import numpy as np
import scipy

# from mpi4py import MPI


def nuclear_error(A, A_nyst, k):
    # TODO compute error of rank-k truncation of the Nystroem approx. using the nuclear norm

    return


def rand_nystrom_seq(
    A: np.ndarray, Omega: np.ndarray, return_extra: bool = True
) -> np.ndarray:

    C = A @ Omega
    B = Omega.T @ C

    try:
        # Try Cholesky
        L = np.linalg.cholesky(B)
        Z = np.linalg.lstsq(L, C.T)[0]
        Z = Z.T
    except np.linalg.LinAlgError as err:
        # Do LDL Factorization
        lu, d, perm = scipy.linalg.ldl(B)
        # Question for you: why is the following line not 100% correct?
        lu = lu @ np.sqrt(np.abs(d))
        # Does this factorization actually work?
        L = lu[perm, :]
        Cperm = C[:, perm]
        Z = np.linalg.lstsq(L, np.transpose(Cperm))[0]
        Z = np.transpose(Z)

    Q, R = np.linalg.qr(Z)
    U_tilde, S, V = np.linalg.svd(R)
    Sigma = np.diag(S)
    U_hat = Q @ U_tilde

    if return_extra:
        S_B = np.linalg.cond(B)
        rank_A = np.linalg.matrix_rank(A)
        return U_hat, Sigma @ Sigma, U_hat.T, S_B, rank_A
    else:
        return U_hat, Sigma @ Sigma, U_hat.T


def create_sketch_matrix_gaussian(n: int, l: int) -> np.ndarray:
    return np.random.normal(loc=0.0, scale=1.0, size=[n, l])
