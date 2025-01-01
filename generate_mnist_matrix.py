import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from data_helpers import get_MNIST_data

if __name__ == "__main__":

    # Pre-requisite: file "mnist.scale" with path: "MATH-505_project-2/data/mnist.scale".
    # The .npy files will be saved in the same folder
    # Images of the matrix will be stored in "results/"
    # Better to run with option "seqential" except when n becomes too big (threshold depends on the machine)

    n = 2**10  # Choose the matrix size!
    method = "sequential"  # "sequential", "vectorized"
    c = 100

    FILE_NAME = "data/mnist.scale"
    print("Getting the data...")
    A3 = get_MNIST_data(FILE_NAME, n=n, c=c, method=method)
    print("Is SPD: ", np.all(np.linalg.eigvals(A3) > 0))
