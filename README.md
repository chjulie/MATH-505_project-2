# Randomized Nyström low rank approximation

Second project for the class MATH-505 "HPC for Numerical Methods and Data Analysis".

## Repository Structure 

**Helpers**:
- `main_nystrom.py`: test the nyström algorithm (sequential and parallelized) for a specific matrix (choose at the beginning of the file)
- `functions.py`: all functions needed by the main files, including: 
    - Sequential and parallel Nyström algorithm
    - Fast Walsh-Hadamard Transform (vectorized or non-vectorized function)
    - Relative nuclear norm computation
    - TSQR
- `generate_mnist_matrix.py`: main to generate a matrix from the MNIST dataset with a specified size
- `data_helpers.py`: functions to generate the matrices used in this project
- `visualizations_pol_exp_matrices.py`: produce visualizations of the exponential and polynomial decay matrices

**Main Scripts**:

*Stability Analysis*
- `stability_analysis.py`: test stability of Nyström algorithm with the parallelized algorithm for varying l (for exponential and polynomial decay matrices)
- `stability_analysis_parallel.py`: test stability of Nyström algorithm with the parallelized algorithm. It will be ran a few times with a different number of processors P, the goal being to check the stability of the algorithm as P increases
- `stability_analysis_sequential.py`: same as `stability_analysis_parallel.py` but for the sequential Nyström algorithm
- `stability_analysis_plot.py`: plot the results of stability_analysis_parallel.py and `stability_analysis_sequential.py` for P = 1, 4, 16, 64

*Runtime Analysis*
- `runtimes_parallel.py`: calculate runtimes of the parallel Nyström algorithm
- `runtimes_sequential.py`: caluculate runtimes of the sequential Nystrôm algorithm for different sizes and sketching dimensions
- `runtimes_analysis.py`: plot runtimes obtained with `runtimes_parallel.py` and `runtimes_sequential.py`

## Authors

- Julie Charlet, MSc in Computational Science and Engineering at EPFL
- Mathilde Simoni, MSc in Computational Science and Engineering at EPFL