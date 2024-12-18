import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# from mpi4py import MPI

from data_helpers import (
    pol_decay,
    exp_decay,
    get_MNIST_data,
)

# from functions import nuclear_error, random_nystroem, p_random_nystroem

if __name__ == "__main__":

    # 1. Import datasets
    # 1.1 Synthetic dataset (polynomial and exponential decay matrices)
    # (ex 9)

    n = 100  # matrix dimension
    Rs = [5, 10, 20]  # effective rank
    ps = [0.5, 1, 2]  # controls the rate of polynomial decay
    qs = [0.1, 0.25, 1.0]  # controls the rate of exponential decay

    # A1 = pol_decay(n, Rs[0], ps[0])
    # A2 = exp_decay(n, Rs[0], qs[0])

    # 1.2 MNIST dataset
    # * Pour Mathilde: * <3
    # run le main en changeant la valeur de n (les noms des fichiers se font automatiquement donc pas besoin d'aller changer la fonction)
    # pre-requisite: il te faut le fichier "mnist.scale" storer comme ça: "MATH-505_project-2/data/mnist.scale".
    # Les fichiers .npy se sauveront dans le même dossier
    # les images des matrices vont dans "results"
    # Perso j'ai testé n=8, n=256 et n=int(2**13).
    # run avec method="vectorized" pour les 2 premier et "sequential" si tu as plus de place
    FILE_NAME = "data/mnist.scale"
    # A3 = get_MNIST_data(FILE_NAME, n=1024, c=100, method="vectorized")
    # print("Is SPD: ", np.all(np.linalg.eigvals(A3) > 0))

    A3 = np.load("data/mnist_" + str(1024) + ".npy")
    print("Is SPD: ", np.all(np.linalg.eigvals(A3) > 0))
    print(np.linalg.eigvals(A3)[:100])
    plt.plot(range(1024), np.linalg.eigvals(A3))
    plt.show()

    # 2. Investigation of numerical stability of randomized Nystroem

    # 4. Sequential runtimes of of randomized Nystroem

    # 5. Parallel performance of randomized Nystroem
