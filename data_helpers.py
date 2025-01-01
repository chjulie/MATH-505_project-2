import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


def pol_decay(n: int, R: int, p: float) -> np.array:

    d1 = np.ones(R, dtype=float)
    d2 = np.arange(n - R, dtype=float)

    f = lambda i, p: (i + 2) ** (-p)

    d2_fin = f(d2, p)
    d = np.concatenate((d1, d2_fin))
    A = np.diag(d)

    return A


def exp_decay(n: int, R: int, q: float) -> np.array:

    d1 = np.ones(R)
    d2 = np.arange(n - R)

    f = lambda i, q: 10 ** (-(i + 1) * q)

    d2_fin = f(d2, q)
    d = np.concatenate((d1, d2_fin))
    A = np.diag(d)

    return A


def RBF(x, c):
    l2_norm_difference = np.linalg.norm(x, axis=2)
    rbf = np.exp(-(l2_norm_difference**2) / c**2)

    return rbf


def read_data(filename):
    data = pd.read_csv(filename)


def get_MNIST_data(filename: str, n: int, c: float, method: str) -> np.array:
    _ = np.newaxis
    start_time = time.time()

    # read file
    data = pd.read_csv("data/mnist.scale")
    # check value of 1000
    x_data = np.zeros((n, 1000))

    inter_time_1 = time.time()

    # parse data
    for line in range(n):
        # isol
        line_str = data.iloc[line, :][0]
        rowid_data_pairs = line_str[1:].split()

        # split ids and values
        ids = [int(pair.split(":")[0]) for pair in rowid_data_pairs]
        values = [float(pair.split(":")[1]) for pair in rowid_data_pairs]
        x_data[line, ids] = values

    # Maske sure that it is normalized
    inter_time_2 = time.time()

    # generate A using radial basis function
    if method == "vectorized":
        difference_array = x_data[:, _] - x_data[_, :]
        print("difference array shape: ", difference_array.shape)
        A = RBF(difference_array, c)
        # A = np.exp(- difference_array ** 2 / c**2)
        print("A shape: ", A.shape)
    elif method == "sequential":
        A = np.zeros((n, n))
        for j in range(n):
            if j % 500 == 0:
                print(j)
            for i in range(j):
                A[i, j] = np.exp(
                    -np.linalg.norm(x_data[i, :] - x_data[j, :]) ** 2 / c**2
                )

        A = A + np.transpose(A)
        for i in range(n):
            A[i, i] = 1
        # np.fill_diagonal(A, 1.0)
    else:
        print("Invalid method")

    inter_time_3 = time.time()

    # save array in .npy format (most efficient for numerical data)
    npy_file_name = filename[:-6] + "_" + str(n) + ".npy"
    np.save(npy_file_name, A)

    inter_time_4 = time.time()

    # visualize generated matrix
    fig, ax = plt.subplots()
    cmap = "inferno"
    im = ax.imshow(A, cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax, cmap=cmap)
    # cbar.ax.set_ylabel(rotation=-90, va="bottom")
    ax.set_title(
        f"Visualization of matrix A generated with RBF\nfrom the MNIST dataset, N = {n}"
    )
    plt.savefig(f"results/A_MNIST_{n}_visualization.png", bbox_inches="tight")

    final_time = time.time()

    print(f"Total run time: {final_time-start_time:.3f}")
    print(f"> Load MNIST data: {inter_time_1-start_time:.3f}")
    print(f"> Parse data: {inter_time_2-inter_time_1:.3f}")
    print(f"> Generate A with RBF: {inter_time_3-inter_time_2:.3f}")
    print(f"> Save matrix A: {inter_time_4-inter_time_3:.3f}")
    print(f"> Visualize matrix A: {final_time-inter_time_4:.3f}")

    return A
