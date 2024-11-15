import numpy as np
import matplotlib.pyplot as plt

def LR_PSD(n: int, R: int, xi: float) -> np.array:

    Lr = np.zeros((n,n))
    Lr[:R, :R] = np.identity(R)

    G = np.random.standard_normal(n**2).reshape((n,n))
    W = G @ G.T 

    A = Lr + np.multiply(xi/n, W)

    return A

def pol_decay(n: int, R: int, p: float) -> np.array:

    d1 = np.ones(R, dtype=float)
    d2 = np.arange(n-R, dtype=float)

    f = lambda i,p: (i+2) ** (-p) 

    d2_fin = f(d2, p)
    d = np.concatenate((d1, d2_fin))
    A = np.diag(d)

    return A

def exp_decay(n: int, R: int, q: float) -> np.array:

    d1 = np.ones(R)
    d2 = np.arange(n-R)

    f = lambda i,q: 10**(-(i+1)*q)

    d2_fin = f(d2, q)
    d = np.concatenate((d1, d2_fin))
    A = np.diag(d)

    return A

def nystroem(A: np.array, Omega: np.array) -> np.array:

    C = A @ Omega
    B = Omega.T @ C
    L = np.linalg.cholesky(B)
    Z = np.linalg.solve(C, L.T)
    Q, R = np.linalg.qr(Z)
    U_tilde, S, V = np.linalg.svd(R)
    U_hat = Q @ U_tilde
    A_nyst = U_hat @ np.linalg.matrix_power(S,2) @ U_hat.T

    return A_nyst, B

if __name__ == "__main__":

    ## Exercise 1: Create test matrices
    N = 100
    Rs = [5, 10, 20]

    # Low-rank and PSD noise
    xis = [1e-4, 1e-2, 1e-1]
    # A = LR_PSD(N, Rs[0], xis[0])

    # Polynomial decay
    ps = [0.5, 1, 2]
    # B = pol_decay(N, Rs[0], ps[0])

    # Exponential decay
    qs = [0.1, 0.25, 1]
    # C = exp_decay(N, Rs[0], qs[0])

    ## Exercises 2: Randomized Nyström
    # a. Plot the singualr values of the test matrices
    x = np.arange(0, N)

    fig, ax = plt.subplots(figsize=(12,4))

    for i in range(3):
        A = LR_PSD(N, Rs[0], xis[i])
        _, S_a, _ = np.linalg.svd(A)
        label = r"$\xi$: " + str(xis[i])
        ax.plot(x, S_a, label=label)
        ax.grid(True)
        ax.set_title('Low-rank and PSD noise')
        ax.legend()

    plt.savefig('plots/SV_test_A.png', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(12,4))

    for i in range(3):
        B = pol_decay(N, Rs[0], ps[i])
        _, S_b, _ = np.linalg.svd(B)
        label = r"$p$: " + str(ps[i])
        ax.plot(x, S_b, label=label)
        ax.grid(True)
        ax.set_title('Polynomial decay')
        ax.legend()

    plt.savefig('plots/SV_test_B.png', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(12,4))


    for i in range(3):
        C = exp_decay(N, Rs[0], qs[0])
        _, S_c, _ = np.linalg.svd(C)
        label = r"$q$: " + str(qs[i])
        ax.plot(x, S_c, label=label)
        ax.grid(True)
        ax.set_title('Exponential decay')
        ax.legend()

    plt.savefig('plots/SV_test_C.png', bbox_inches='tight')

    #d. Plot the singular values of B for each of the tests matrices
