import numpy as np
import pandas as pd


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

def RBF(x, c):
    l2_norm_difference = np.linalg.norm(x)
    rbf = np.exp(- l2_norm_difference ** 2 / c**2)

    return rbf

def read_data(filename):
    data = pd.read_csv(filename)
    

def get_MNIST_data(filename: str, n: int, c: float) -> np.array:
    #TODO: Build n,n dense matrix A from MNIST dataset using radial basis function
    _ = np.newaxis

    # read file
    data = mmread(filename)
    data_x = data[:n, :n]

    # Maske sure that it is normalized

    difference_array = data_x[:,_] - data_x[_,:]
    A = RBF(data, c)

    # save array in .npy format (most efficient for numerical data)
    npy_file_name = 'array_' + filename[:-6] + '.npy'
    np.save(npy_file_name, A)

    # Save data using pickle format

    return A.shape

def get_YearpredictionMSD_data() -> np.array:
    #TODO: Build n,n dense matrix A from YearpredictionMSD dataset using radial basis function
    c = 1e4 # or 1e5


    # Normalize data

    return A

