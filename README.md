# Multivariate Regression and Classification Using an Adaptive Neuro-Fuzzy Inference System (Takagi-Sugeno) and Metaheuristic Optimization

There are three versions:

- [Readme](./Code_Cpp_Vanilla/README.md) for the vanilla version. All arrays and matrices are dynamically allocated using *new* and all the functions/templates needed for the basic array-matrix operations have been coded.

- [Readme](./Code_Cpp_Eigen/README.md) for the Eigen version. All the basic array-matrix operations are managed by the Eigen library.

- [Readme](./Code_Cpp_PSO/README.md) for the version using particle swarm optimization (and the Eigen library) instead of simulated annealing optimization.

## References

- Mathematical background from [Neuro-Fuzzy and Soft Computing](https://ieeexplore.ieee.org/document/633847), by Jang, Sun, and Mizutani.

- Datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php).

- [Git code](https://github.com/gabrielegilardi/ANFIS) for the Python version of the ANFIS.

- [Git code](https://github.com/gabrielegilardi/SimulatedAnnealing) for the C++ version of the Simulated Annealing Optimizer.

- [Git code](https://github.com/gabrielegilardi/PSO) for the Python version of the Particle Swarm Optimizer.

- [Eigen](https://eigen.tuxfamily.org/) template library for linear algebra.
