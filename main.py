import numpy as np 
import matplotlib.pyplot as pyplot
from mpi4py import MPI
from functions import random_nystroem, p_random_nystroem

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

def MNIST_data() -> np.array:
    #TODO: Build n,n dense matrix A from MNIST dataset using radial basis function
    c = 100

    return A

def YearpredictionMSD_data() -> np.array:
    #TODO: Build n,n dense matrix A from YearpredictionMSD dataset using radial basis function
    c = 1e4 # or 1e5

    return A

def nuclear_error(A, A_nyst, k):
    #TODO compute error of rank-k truncation of the Nystroem approx. using the nuclear norm

    return

if __name__ == "__main__:

    # 1. Import datasets
    # 1.1 Synthetic dataset (polynomial and exponential decay matrices)
    # (ex 9)

    n = 100 # matrix dimension
    Rs = [5, 10, 20] # effective rank
    ps = [0.5, 1, 2] # controls the rate of polynomial decay
    qs = [0.1, 0.25, 1.0] # controls the rate of exponential decay

    A1 = pol_decay(n, R, p)
    A2 = exp_decay(n, R, q)

    # 1.2.1 MNIST dataset
    # (ex 10)


    # 1.2.2 YearpredictionMSD dataset


    # 2. Investigation of numerical stability of randomized Nystroem


    # 4. Sequential runtimes of of randomized Nystroem


    # 5. Parallel performance of randomized Nystroem


