# Randomized Nyström low rank approximation

Project 2 for the class MATH-505 HPC for numerical methods and data analysis

## Structure 

- main_nystrom.py: test the nyström algorithm (sequential and parallelized) for a specific matrix (choose at the beginning of the file)
- functions.py: all functions needed by the main files, including: 
    - Sequential and parallel Nyström algorithm
    - Fast Walsh-Hadamard Transform (vectorized or non-vectorized function)
    - Nuclear norm computation
    - TSQR
- stability_analysis_sequential.py: test stability of Nyström algorithm with the sequential algorithm
- stability_analysis_parallel.py: test stability of Nyström algorithm with the parallelized algorithm
- runtimes.py
- generate_mnist_matrix.py: generate a matric from the MNIST dataset with a specified size
- data_helpers.py: functions to generate the matrices used in this project

