# Randomized Nyström low rank approximation

Project 2 for the class MATH-505 HPC for numerical methods and data analysis

## Structure 

Helpers:
- main_nystrom.py: test the nyström algorithm (sequential and parallelized) for a specific matrix (choose at the beginning of the file)
- functions.py: all functions needed by the main files, including: 
    - Sequential and parallel Nyström algorithm
    - Fast Walsh-Hadamard Transform (vectorized or non-vectorized function)
    - Nuclear norm computation
    - TSQR
- generate_mnist_matrix.py: generate a matric from the MNIST dataset with a specified size
- data_helpers.py: functions to generate the matrices used in this project
- visualizations_pol_exp_matrices.py: produce visualizations of the exponential and polynomial decay matrices

Main Scripts:

*Stability Analysis*
- stability_analysis.py: test stability of Nyström algorithm with the parallelized algorithm for varying l (for exponential and polynomial matrices)
- stability_analysis_parallel.py: test stability of Nyström algorithm with the parallelized algorithm for MNIST matrix => will be ran mny times with a different number of processors P, the goal being to check the stability of the algorithm as P increases
- stability_analysis_sequential.py: same as stability_analysis_parallel.py but for sequential Nyström algorithm
- stability_analysis_plot.py: plot the results of stability_analysis_parallel.py for different P = 1, 4, 16, 64

*Runtime Analysis*
- runtimes_analysis.py
- runtimes_parallel.py
- runtimes_sequential.py

