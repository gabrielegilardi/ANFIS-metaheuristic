# Multivariate Regression and Classification Using an Adaptive Neuro-Fuzzy Inference System (Takagi-Sugeno) and Simulated Annealing Optimization

## Features

- The code has been written in plain vanilla C++ and tested using g++ 11.2.0 in MinGW-W64 9.0.0-r1.
- Multi-input/multi-output (multivariate) adaptive neuro-fuzzy inference system (ANFIS) implementation for regression and classification.
- Quadratic cost function for continuous problems and cross-entropy cost function for classification problems.
- Classes in classification problems can be determined automatically.
- Sigmoid and cross-entropy function are computed using a numerically stable implementation.
- Generalized Bell curves depending on three parameters (mean, standard deviation, and exponent) are used as premise membership functions.
- Hyperplanes depending on the number of features are used as consequent functions.
- A population-based simulated annealing optimizer (SA) is used to solve the minimization problem.
- Limits/constraints on the parameter values (similar to regularization in neural networks) can be easily done through the SA boundary arrays.
- The `ANFIS` class is not constrained to the SA solver but it can be easily adapted to any other optimizer not gradient-based.
- Files *utils.cpp* and *template.h* consist of several helper functions.
- Included are also helper functions to build the SA boundary arrays and to build classes in classifications problems.
- Usage: *test.exe example*.

## Main Parameters

`example` Name of the example to run (plant, stock, wine, pulsar.)

`classification` Defines the type of problem, with `true` specifying a classification problem.

`split_factor` Split value between training and test data.

`data_file` File name with the dataset (comma separated format).

`n_mf` Array with the number of premise functions of each feature. Its lenght must be the same as the number of features.

`nPop`, `epochs` Number of agents (population) and number of iterations.

`mu_delta` Allowed variation (plus/minus) of the mean in the premise functions. It is given as fraction of the corresponding feature data range.

`s_par` Center value and allowed variation (plus/minus) of the standard deviation in the premise functions. The center value is scaled based on the corresponding feature data range.

`c_par` Range of allowed values of the exponent in the premise functions.

`A_par` Range of allowed values of the coefficients in the consequent functions.

`tol` Tolerance used to group classes in classification problems.

`agents` Array of agents used by the SA solver. Each agent is one ANFIS instance.

See file *sa.cpp* for the meaning of the other quantities defined in structure `p`.

## Examples

There are four examples in *test.cpp*: plant, stock, wine, pulsar.

**plant** (single-label regression problem)

- The dataset has 4 features (inputs), 1 label (output), and 9568 samples.

- The ANFIS has a layout of [1, 1, 1, 1] and 17 variables.

- Predicted/actual correlation values: 0.964 (training), 0.963 (test).

- Original dataset: <https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant>.

**stock** (multi-label regression problem)

- The dataset has 3 features (inputs), 2 labels (outputs), and 536 samples.

- The ANFIS has a layout of [2, 2, 2] and 82 variables.

- Predicted/actual correlation values: 0.891 (training), 0.863 (test).

- Original dataset: <https://archive.ics.uci.edu/ml/datasets/ISTANBUL+STOCK+EXCHANGE>.

**wine** (Multi-class classification problem)

- The dataset has 2 features (inputs), 6 classes (outputs), and 1599 samples.

- The ANFIS has a layout of [3, 2] and 123 variables.

- Predicted/actual accuracy values: 55.7% (training), 55.8% (test).

- Original dataset: <https://archive.ics.uci.edu/ml/datasets/Wine+Quality>.

**pulsar** (Multi-class classification problem)

- The dataset has 3 features (inputs), 2 classes (outputs), and 17898 samples.

- The ANFIS has a layout of [3, 4, 2] and 219 variables.

- Predicted/actual accuracy values: 97.7% (training), 97.9% (test).

- Original dataset: <https://archive.ics.uci.edu/ml/datasets/HTRU2>.

## References

- Mathematical background from [Neuro-Fuzzy and Soft Computing](https://ieeexplore.ieee.org/document/633847), by Jang, Sun, and Mizutani.

- Datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php)".

- [Git code](https://github.com/gabrielegilardi/ANFIS) for the Python version of the ANFIS.

- [Git code](https://github.com/gabrielegilardi/SimulatedAnnealing) for the C++ version of the Simulated Annealing Optimizer.
