import numpy as np 
import matplotlib.pyplot as pyplot
import pandas as pd
#from mpi4py import MPI

from data_helpers import pol_decay, exp_decay, get_MNIST_data, get_YearpredictionMSD_data
#from functions import nuclear_error, random_nystroem, p_random_nystroem

if __name__ == "__main__":

    # 1. Import datasets
    # 1.1 Synthetic dataset (polynomial and exponential decay matrices)
    # (ex 9)

    n = 100 # matrix dimension
    Rs = [5, 10, 20] # effective rank
    ps = [0.5, 1, 2] # controls the rate of polynomial decay
    qs = [0.1, 0.25, 1.0] # controls the rate of exponential decay

    # A1 = pol_decay(n, Rs[0], ps[0])
    # A2 = exp_decay(n, Rs[0], qs[0])

    # 1.2.1 MNIST dataset
    # (ex 10)

    data = pd.read_csv('data/mnist.scale')
    r = 45
    col_data_str = data.iloc[r,:][0]

    #
    col_index = col_data_str[0]
    rowid_data_pairs = col_data_str[1:].split()

    print(len(rowid_data_pairs))

    #size = get_MNIST_data('data/mnist.scale', 10, 2)

    # 1.2.2 YearpredictionMSD dataset


    # 2. Investigation of numerical stability of randomized Nystroem


    # 4. Sequential runtimes of of randomized Nystroem


    # 5. Parallel performance of randomized Nystroem


