import numpy as np
import scipy
import random
import math
import time

# import torch
# from hadamard_transform import hadamard_transform
from scipy.linalg import hadamard
from typing import Tuple
from mpi4py import MPI


def nuclear_error(A, U, Sigma):
    # TODO compute error of rank-k truncation of the Nystroem approx. using the nuclear norm
    num = np.linalg.norm(A - ..., ord='nuc')

    return

def rand_nystrom_sequential(
    A: np.ndarray, 
    seed: int, 
    n: int,
    sketching: str, 
    k: int, 
    l: int,
    return_extra: bool = False
) -> np.ndarray:

    np.random.seed(seed)

    if sketching == "SHRT":
        # C = Ω × A
        d = np.array([1 if np.random.random() < 0.5 else -1 for _ in range(n)])
        C = np.multiply(np.sqrt(n / l) * d[:,np.newaxis], A)
        C = np.array([FWHT(C[:,i]) for i in range(n)]).T
        R = random.sample(range(n), l)
        C = C[R,:]

        # B = Ω × C.T
        B = np.multiply(np.sqrt(n / l) * d[:,np.newaxis], C.T)
        B = np.array([FWHT(B[:,i]) for i in range(l)]).T
        B = B[R,:]
    
    elif sketching == "gaussian":

        Omega = np.random.normal(loc=0.0, scale=1.0, size=[l,n])
        C = Omega @ A
        B = Omega @ C.T

    else:
        raise (NotImplementedError)


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
        return U_hat_k, Sigma_k @ Sigma_k, S_B, rank_A
    else:
        return U_hat_k, Sigma_k @ Sigma_k


def rand_nystrom_parallel(
    A_local: np.ndarray,
    Omega_local: np.ndarray,
    OmegaT_local: np.ndarray,
    n: int,
    k: int,
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

def SFWHT(a):
    """ Fast Walsh–Hadamard Transform of vector a
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

def FWHT(x):
    """ Fast Walsh-Hadamard Transform
    Based on mex function written by Chengbo Li@Rice Uni for his TVAL3 algorithm.
    His code is according to the K.G. Beauchamp's book -- Applications of Walsh and Related Functions.

    credits: https://github.com/dingluo/fwht
    """
    x = x.squeeze()
    N = x.size
    G = int(N/2) # Number of Groups
    M = 2 # Number of Members in Each Group

    # First stage
    y = np.zeros((int(N/2),2))
    y[:,0] = x[0::2] + x[1::2]
    y[:,1] = x[0::2] - x[1::2]
    x = y.copy()
    
    # Second and further stage
    for nStage in range(2,int(math.log(N,2))+1):
        y = np.zeros((int(G/2),int(M*2)))
        G = int(G)
        M = int(M)
        y[0:int(G/2),0:int(M*2):4] = x[0:G:2,0:M:2] + x[1:G:2,0:M:2]
        y[0:int(G/2),1:int(M*2):4] = x[0:G:2,1:M:2] + x[1:G:2,1:M:2]
        y[0:int(G/2),2:int(M*2):4] = x[0:G:2,0:M:2] - x[1:G:2,0:M:2]
        y[0:int(G/2),3:int(M*2):4] = x[0:G:2,1:M:2] - x[1:G:2,1:M:2]
        x = y.copy()
        G = G/2
        M = M*2
    x = y[0,:]
    x = x.reshape((x.size,1)).squeeze(-1)
    return x / math.sqrt(N)

def rand_nystrom_parallel_SHRT(
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
    rank,
    rank_cols,
    rank_rows,
    size_cols,
) -> Tuple[np.ndarray, np.ndarray]:

    t1 = time.time()

    if sketching == "SHRT":
        # share the seed amongst rows (distribute Ω over the columns)
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
        C_local = np.array([FWHT(C_local[:,i]) for i in range(n_local)]).T
        # A = R @ A
        # use global seed to select rows
        random.seed(seed_global)
        R = random.sample(range(n_local), l)
        C_local = C_local[R, :]

        # Compute C = DL R H DR A
        C_local = np.multiply(dl[:, np.newaxis], C_local)

        # print(f" * 1. Rank {rank}, rank_cols: {rank_cols}, rank_rows: {rank_rows}: C_local: {C_local} \n")
        # column-wise sum-reduce => use comm_cols for communication in between rows??
        # C_cols = comm_rows.allreduce(C_local, op=MPI.SUM)
        comm_rows.Reduce(C_local, C, op=MPI.SUM, root=0)

        # print(f" * 2. Rank {rank}, rank_cols: {rank_cols}, rank_rows: {rank_rows}: C_cols: {C_cols}\n")
        # 2.1 Compute B = Ω × C.T
        B = None
        if rank_cols == 0:
            B = np.empty((l, l), dtype="float")

        if rank_rows == 0:
            # Apply Ω matrix
            # share the seed amongst columns (distribute Ω over the rows)
            seed_local = rank_cols
            np.random.seed(seed_local)
            dr = np.array([1 if np.random.random() < 0.5 else -1 for _ in range(n_local)])
            dl = np.array([1 if np.random.random() < 0.5 else -1 for _ in range(l)])

            B_local = np.multiply(np.sqrt(n_local / l) * dr[:, np.newaxis], C.T)
            # !! list comprehension construction swaps axis 0 and 1!!
            B_local = np.array([FWHT(B_local[:,i]) for i in range(l)]).T # only l columns instead of n_local now
            B_local = B_local[R, :]
            B_local = np.multiply(dl[:, np.newaxis], B_local)

            # B = comm_cols.allreduce(B_local, op=MPI.SUM)
            # comm.Reduce(B_local, B, op=MPI.SUM, root=0)
            comm_cols.Reduce(B_local, B, op=MPI.SUM, root=0)

            if rank == 0:
                print(' B.shape: ', B.shape)
                print(' C.shape: ', C.shape)

            # print(f" * Rank {rank}, rank_cols: {rank_cols}, rank_rows: {rank_rows}: B: {B}\n")

    elif sketching == "gaussian":

        np.random.seed(rank_rows)

        C = None
        if rank_rows == 0:
            C = np.empty((l, n_local), dtype="float")

        # Generate gaussian matrix
        C_local = None
        C_cols = None
        C_local = np.random.normal(loc=0.0, scale=1.0, size=[l, n_local]) @ A_local
        # if rank == 0:
            # print(f" * Ω.shape 1: ", (np.random.normal(loc=0.0, scale=1.0, size=[l, n_local])).shape)
            # print(f" * A_local.shape: ", A_local.shape)

        # print(f" * Rank {rank}, rank_cols: {rank_cols}, rank_rows: {rank_rows}: C_local: {C_local}\n")
        # C_cols = comm_rows.allreduce(C_local, op=MPI.SUM)
        comm_rows.Reduce(C_local, C, op=MPI.SUM, root=0)
        # print(f" * Rank {rank}, rank_cols: {rank_cols}, rank_rows: {rank_rows}: C: {C}\n")

        B = None
        if rank_cols == 0:
            B = np.empty((l, l), dtype="float")

        if rank_rows == 0:
    
            # Apply Ω matrix
            # share the seed amongst columns (distribute Ω over the rows)
            np.random.seed(rank_cols)
            # if rank == 0:
            #     print(f" * Ω.shape 2: ", (np.random.normal(loc=0.0, scale=1.0, size=[l, n_local])).shape)
            #     print(f" * C_cols^T.shape: ", (C.T).shape)

            B_local = np.random.normal(loc=0.0, scale=1.0, size=[l, n_local]) @ C.T
            # B = comm_cols.allreduce(B_local, op=MPI.SUM)
            comm_cols.Reduce(B_local, B, op=MPI.SUM, root=0)
            # print(f" * Rank {rank}, rank_cols: {rank_cols}, rank_rows: {rank_rows}: B: {B}\n")

    else:
        raise (NotImplementedError)
    
    t2 = time.time()
    # comm.Reduce(B_local, B, op=MPI.SUM, root=0)

    L = None
    permute = False
    perm = None
    # 2.2 Compute the Cholesky factorization of B: B = LL^T or eigen value decomposition
    if rank == 0:  # compute only at the root
        try:  # Try Cholesky
            L = np.linalg.cholesky(B)
            print(" > Cholesky succeeded!")
        except np.linalg.LinAlgError as err:
            # Do LDL Factorization (TODO: might need to do change)
            lu, d, perm = scipy.linalg.ldl(B)
            # Question for you: why is the following line not 100% correct?
            lu = lu @ np.sqrt(np.abs(d))
            # Does this factorization actually work?
            L = lu[perm, :]
            permute = True
            print(" > LDL factorization succeeded!")


    L = comm_cols.bcast(L, root=0)  # broadcast through columns
    perm = comm_cols.bcast(perm, root=0)
    if permute == True and rank_rows == 0:
        C_cols = C_cols[:, perm]

    t3 = time.time()

    # 3. Compute Z = C @ L.-T with substitution
    # this is only computed in processes of the first row (with rank_rows = 0)
    Z_local = None
    if rank_rows == 0:
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
        
        # Sigma = np.diag(S)
        # Sigma = Sigma[:k, :k]
        # Sigma_2 = Sigma @ Sigma

    U_tilde = comm_cols.bcast(U_tilde, root=0)  # broadcast through rows

    # 6. Compute U_hat = Q @ U
    # U_hat_local = np.empty((A_local.shape[0], k), dtype="float")
    U_hat_local = None
    if rank_rows == 0:
        # print('Q_local: ', Q_local.shape)
        # print('U_tilde: ', U_tilde.shape)
        U_hat_local = Q_local @ U_tilde
        # print(f" * Rank {rank}, rank_cols: {rank_cols}, rank_rows: {rank_rows}: U_local: {U_hat_local}\n")

    t6 = time.time()

    # PRINT OUT COMPUTATION TIMES
    if rank == 0:
        print("\n ** COMPUTATION TIMES ** \n")
        print(f" - Apply Ω: B = Ω (Ω A).T: {t2-t1:.4f} s.")
        print(f" - Cholesky decomposition: B = L L.T: {t3-t2:.4f} s.")
        print(f" - Z with substitution: Z = C @ L.-T: {t4-t3:.4f} s.")
        print(f" - QR factorization: {t5-t4:.4f} s.")
        print(f" - Truncated rank-r SVD: {t6-t5:.4f} s.\n")

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
    # use local seed to compute diagonal matrices
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
