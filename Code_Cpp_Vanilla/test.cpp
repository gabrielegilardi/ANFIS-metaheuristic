/*
Multivariate Regression and Classification Using an Adaptive Neuro-Fuzzy
Inference System (Takagi-Sugeno) and Simulated Annealing Optimization.

Copyright (c) 2021 Gabriele Gilardi


Features
--------
- The code has been written in plain vanilla C++ and tested using g++ 11.2.0
  in MinGW-W64 9.0.0-r1.
- Multi-input/multi-output (multivariate) adaptive neuro-fuzzy inference
  system (ANFIS) implementation for regression and classification.
- Quadratic cost function for continuous problems and cross-entropy cost
  function for classification problems.
- Classes in classification problems can be determined automatically.
- Sigmoid and cross-entropy function are computed using a numerically stable
  implementation.
- Generalized Bell curves depending on three parameters (mean, standard
  deviation, and exponent) are used as premise membership functions.
- Hyperplanes depending on the number of features are used as consequent
  functions.
- A population-based simulated annealing optimizer (SA) is used to solve the
  minimization problem.
- Limits/constraints on the parameter values (similar to regularization in
  neural networks) can be easily done through the SA boundary arrays.
- The <ANFIS> class is not constrained to the SA solver but it can be easily
  adapted to any other optimizer not gradient-based.
- Files "utils.cpp" and "template.h" consist of several helper functions.
- Included are also helper functions to build the SA boundary arrays and to
  build classes for classifications problems.
- Usage: test.exe <example>.

Main Parameters
---------------
example = plant, stock, wine, pulsar
    Name of the example to run.
classification = true, false
    Defines the type of problem, with <true> specifying a classification problem. 
0 < split_factor < 1
    Split value between training and test data.
data_file
    File name with the dataset (comma separated format).
n_mf
    Array with the number of premise functions of each feature. Its lenght
    must be the same as the number of features.
nPop >=1, epochs >= 1
    Number of agents (population) and number of iterations.
mu_delta >= 0
    Allowed variation (plus/minus) of the mean in the premise functions. It is
    given as fraction of the corresponding feature data range.
s_par > 0
    Center value and allowed variation (plus/minus) of the standard deviation in
    the premise functions. The center value is scaled based on the corresponding
    feature data range.
c_par > 0
    Range of allowed values of the exponent in the premise functions.
A_par
    Range of allowed values of the coefficients in the consequent functions.
tol > 0
    Tolerance used to group classes in classification problems.
agents
    Array of agents used by the SA solver. Each agent is one ANFIS instance. 

See https://github.com/gabrielegilardi/SimulatedAnnealing for the meaning of the
other SA quantities defined in structure <Parameters>.

Examples
--------
There are four examples (see code for parameters and results): 
- plant: single-label regression (continuous) problem.
- stock: multi-label regression (continuous) problem.
- wine: multi-class classification problem.
- pulsar: multi-class classification problem.

References
----------
- Mathematical background from Jang, et al., "Neuro-Fuzzy and Soft Computing"
  @ https://ieeexplore.ieee.org/document/633847

- Datasets from the UCI Machine Learning Repository
  @ https://archive.ics.uci.edu/ml/datasets.php

- Population-Based Simulated Annealing Optimizer 
  @ https://github.com/gabrielegilardi/SimulatedAnnealing
*/

/* Headers */
#include <cmath>
#include <fstream>

#include "utils.hpp"
#include "ANFIS.hpp"
#include "templates.hpp"

/* Structure to pass the parameters (and default values) to the SA solver */
struct Parameters {
    int nPop = 20;
    int epochs = 100;
    int nMove = 50;
    double T0 = 0.1;
    double alphaT = 0.99;
    double sigma0 = 0.1;
    double alphaS = 0.98;
    double prob = 0.5;
    bool normalize = false;
    int* IntVar = nullptr;
    int nIntVar = 0;
};

/* Structure to pass the data to the ANFIS */
struct Arguments {
    double** X = nullptr;         // Training dataset inputs/features
    double** Y = nullptr;         // Training dataset labels/outputs (regression)
    int* Yc = nullptr;            // Training dataset classes (classification)
    int n_samples;                // Number of samples training dataset
    AnfisType* agents = nullptr;  // Array of ANFIS agents
};

/* Prototypes (local functions) */
double* sa(double (*func)(double*, int, Arguments), double* LB, double* UB,
           int nVar, Parameters p, Arguments args, mt19937_64& gen);
double interface(double* theta, int idx, Arguments args) ;
double** read_data(string data_file, int& rows, int& cols, bool flip);
void bounds(double** X, int n_samples, int* MFs, int n_inputs, int n_outputs,
            double mu_delta, double s_par[2], double c_par[2], double A_par[2],
            double*& LB, double*& UB);


/* Main */
int main(int argc, char** argv) 
{
    // Default values (common to all examples unless specified)
    bool classification = false;
    double split_factor = 0.70;
    double mu_delta = 0.2;
    double s_par[2] = {0.5, 0.2};
    double c_par[2] = {1.0, 3.0};
    double A_par[2] = {-10.0, 10.0};
    double tol = 1.e-5;
    int seed = 1234567890;

    // Read example to run
    if (argc != 2) {
        printf("\nUsage: test.exe example_name\n\n");
        exit(EXIT_FAILURE);
    }
    string example = argv[1];

    string data_file;
    int n_inputs;
    int* MFs;
    Parameters p;
    mt19937_64 gen(seed);

    // Single-label continuous problem example
    if (example == "plant") {
        // Dataset: 4 features (inputs), 1 label (output), 9568 samples
        // ANFIS: layout of [1, 1, 1, 1], 17 variables
        // Predicted/actual correlation values: 0.964 (training), 0.963 (test)
        // https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
        data_file = "plant_dataset.csv";
        n_inputs = 4;
        MFs = new_Array<int>(n_inputs);
        set_Array(MFs, n_inputs, 1);
    }

    // Multi-label continuous problem example
    else if (example == "stock") {
        // Dataset: 3 features (inputs), 2 labels (outputs), 536 samples
        // ANFIS: layout of [2, 2, 2], 82 variables
        // Predicted/actual correlation values: 0.891 (training), 0.863 (test)
        // https://archive.ics.uci.edu/ml/datasets/ISTANBUL+STOCK+EXCHANGE
        data_file = "stock_dataset.csv";
        n_inputs = 3;
        MFs = new_Array<int>(n_inputs);
        set_Array(MFs, n_inputs, 2);
        // Changed parameters
        p.nPop = 40;
        p.epochs = 500;
        p.T0 = 0.0;
        p.sigma0 = 0.05;
        A_par[0] = -1.0;
        A_par[1] = 1.0;
    }

    // Multi-class classification problem example
    else if (example == "wine") {
        // Dataset: 2 features (inputs), 6 classes (outputs), 1599 samples
        // ANFIS: layout of [3, 2], 123 variables
        // Predicted/actual accuracy values: 55.7% (training), 55.8% (test).
        // https://archive.ics.uci.edu/ml/datasets/Wine+Quality
        data_file = "wine_dataset.csv";
        n_inputs = 2;
        MFs = new_Array<int>(n_inputs);
        MFs[0] = 3;
        MFs[1] = 2;
        // Changed parameters
        p.nMove = 20;
        p.sigma0 = 0.05;
        p.T0 = 0.0;
        classification = true;
    }

    // Multi-class classification problem example
    else if (example == "pulsar") {
        // Dataset: 3 features (inputs), 2 classes (outputs), 17898 samples
        // ANFIS: layout of [3, 4, 2], 219 variables
        // Predicted/actual accuracy values: 97.7% (training), 97.9% (test).
        // https://archive.ics.uci.edu/ml/datasets/HTRU2
        data_file = "pulsar_dataset.csv";
        n_inputs = 3;
        MFs = new_Array<int>(n_inputs);
        MFs[0] = 3;
        MFs[1] = 4;
        MFs[2] = 2;
        // Changed parameters
        p.nPop = 10;
        p.epochs = 50;
        p.nMove = 10;
        classification = true;
    }

    // Wrong example name
    else  {
        printf("\n%s", example.c_str());
        printf("\n--> Example not found.\n\n");
        exit(EXIT_FAILURE);
    }

    // Read (comma-separated) data from a file (note: samples are along the
    // columns, while inputs and outputs are along the rows)
    int n_rows, n_samples;
    double** data = read_data(data_file, n_rows, n_samples, true);

    // Randomly shuffle the indexes and build the shuffled data matrix
    int *idx_shuffle = shuffle(n_samples, gen);
    double** data_shuffle = new_Array<double>(n_rows, n_samples);
    for (int i=0; i<n_samples; i++) {
        for (int j=0; j<n_rows; j++) {
            data_shuffle[j][idx_shuffle[i]] = data[j][i]; 
        }
    }

    // For a classification problem build the class table (classes must always
    // be on the last row)
    int n_classes, n_outputs;
    double* table;
    if (classification) {
        table = build_class_table(data[n_rows-1], n_samples, n_classes, tol);
        n_outputs = 1;
    }

    // In a regression problem the outputs are always in the last rows
    else {
        n_outputs = n_rows - n_inputs;
    }

    // Build the splitted datasets ( training dataset: 0 --> samples_tr-1,
    // test dataset: samples_tr --> n_samples-1)
    int samples_tr = int(nearbyint(split_factor * double(n_samples)));
    int samples_te = n_samples - samples_tr;
    double** X_tr = new_Array<double>(n_inputs, samples_tr);
    double** Y_tr = new_Array<double>(n_outputs, samples_tr);
    double** X_te = new_Array<double>(n_inputs, samples_te);
    double** Y_te = new_Array<double>(n_outputs, samples_te);
    copy_Array(X_tr, n_inputs, samples_tr, data_shuffle, 0, 0);
    copy_Array(Y_tr, n_outputs, samples_tr, data_shuffle, n_inputs, 0);
    copy_Array(X_te, n_inputs, samples_te, data_shuffle, 0, samples_tr);
    copy_Array(Y_te, n_outputs, samples_te, data_shuffle, n_inputs, samples_tr);

    // Normalize inputs (rows) using the training dataset mean and std
    double** Xn_tr = new_Array <double*>(n_inputs);
    double** Xn_te = new_Array <double*>(n_inputs);
    for (int j=0; j<n_inputs; j++) {
        double mu = mean(X_tr[j], samples_tr);
        double sigma = stdev(X_tr[j], samples_tr);
        Xn_tr[j] = normalize(X_tr[j], samples_tr, mu, sigma);
        Xn_te[j] = normalize(X_te[j], samples_te, mu, sigma);
    }

    // For classification get the classes in the training dataset output (the
    // original classes are re-numbered 0 --> n_classes-1)
    int n_out;
    int* Yc_tr;
    double **Ys_tr, **Ys_te;
    if (classification) {
        Yc_tr = get_classes(Y_tr[0], samples_tr, table, n_classes, tol);
        n_out = n_classes;          // Actual number of outputs used by ANFIS
    }

    // For regression scale the outputs (rows) to the interval [-1, 1], to
    // reduce the range of <A> 
    else {
        Ys_tr = new_Array <double*>(n_outputs);
        Ys_te = new_Array <double*>(n_outputs);
        for (int j=0; j<n_outputs; j++) {
            double Y_min = value_min(Y_tr[j], samples_tr);
            double Y_max = value_max(Y_tr[j], samples_tr);
            Ys_tr[j] = scale(Y_tr[j], samples_tr, Y_min, Y_max);
            Ys_te[j] = scale(Y_te[j], samples_te, Y_min, Y_max);
        }
        n_out = n_outputs;
    }

    // Init ANFIS agents
    AnfisType* agents = new_Array<AnfisType>(p.nPop);
    for (int i=0; i<p.nPop; i++) {
        agents[i].init(n_inputs, n_out, MFs);
    }

    // Init arguments to be passed
    Arguments args;
    args.X = Xn_tr;
    if (classification) {
        args.Y = nullptr;
        args.Yc = Yc_tr;
    }
    else {
        args.Y = Ys_tr;
        args.Yc = nullptr;
    }
    args.n_samples = samples_tr;
    args.agents = agents;

    // Init the SA search space boundaries (arrays LB and UB)
    double *LB, *UB;
    bounds(Xn_tr, samples_tr, MFs, n_inputs, n_out, mu_delta, s_par, c_par,
           A_par, LB, UB);

    // Show dataset info
    printf("\n\n===== Dataset info =====");
    printf("\n- Example: %s", example.c_str());
    printf("\n- File: %s", data_file.c_str());
    printf("\n- Total samples: %d", n_samples);
    if (classification) {
        printf("\n- Number of classes: %d", n_classes);
        printf("\n- Class table:");
        for (int i=0; i<n_classes; i++) {
            printf(" %g ", table[i]);
        }
    }
    printf("\n- Training samples: %d", samples_tr);
    printf("\n- Test samples: %d", samples_te);

    // Show ANFIS info
    agents[0].info();
    int n_var = agents[0].n_var;        // Number of variables
    int n_pf = agents[0].n_pf;          // Number of premise functions
    int n_cf = agents[0].n_cf;          // Number of consequent functions

    // Solve
    double (*func)(double[], int, Arguments);
    func = interface;
    double* best_sol = sa(func, LB, UB, n_var, p, args, gen);

    // Re-build the ANFIS with the best solution and evaluate datasets
    double J;
    double **Yp_tr, **Yp_te;
    AnfisType best_agent;
    best_agent.init(n_inputs, n_out, MFs);
    if (classification) {
        J = best_agent.create_model(best_sol, Xn_tr, Yc_tr, samples_tr);
        Yp_tr = best_agent.eval_data(Xn_tr, samples_tr, table);
        Yp_te = best_agent.eval_data(Xn_te, samples_te, table);
    }
    else {
        J = best_agent.create_model(best_sol, Xn_tr, Ys_tr, samples_tr);
        Yp_tr = best_agent.eval_data(Xn_tr, samples_tr);
        Yp_te = best_agent.eval_data(Xn_te, samples_te);
    }
    // Show results
    printf("\n\n===== Results =====");
    printf("\nJ = %g", J);
    printf("\n\n    mu        c         s");
    for (int i=0; i<n_pf; i++) {
        printf("\n%8.4f  %8.4f  %8.4f", best_agent.mu[i], best_agent.s[i],
                                        best_agent.c[i]);
    }
    printf("\n\nA");
    for (int i=0; i<n_inputs+1; i++) {
        printf("\n");
        for (int j=0; j<n_cf*n_out; j++) {
            printf("%10.4f ", best_agent.A[i][j]);
        }
    }
    printf("\n\n\n===== Stats =====");
    if (classification) {
        printf("\nAccuracy (tr) = %g", accuracy(Yp_tr, Y_tr, n_outputs, samples_tr));
        printf("\nAccuracy (te) = %g", accuracy(Yp_te, Y_te, n_outputs, samples_te));
        printf("\nCorrelation (tr) = %g", calc_corr(Yp_tr, Y_tr, n_outputs,samples_tr));
        printf("\nCorrelation (te) = %g", calc_corr(Yp_te, Y_te, n_outputs, samples_te));
    }
    else {
        printf("\nCorrelation (tr) = %g", calc_corr(Yp_tr, Ys_tr, n_outputs, samples_tr));
        printf("\nCorrelation (te) = %g", calc_corr(Yp_te, Ys_te, n_outputs, samples_te));
        printf("\nRMSE (tr) = %g", rmse(Yp_tr, Ys_tr, n_outputs, samples_tr));
        printf("\nRMSE (te) = %g", rmse(Yp_te, Ys_te, n_outputs, samples_te));
    }

    // De-allocate all dynamic arrays and matrices
    del_Array(MFs);
    del_Array(idx_shuffle);
    del_Array(data, n_rows);
    del_Array(data_shuffle, n_rows);
    del_Array(X_tr, n_inputs);
    del_Array(X_te, n_inputs);
    del_Array(LB);
    del_Array(UB);
    del_Array(best_sol);
    del_Array(Y_tr, n_outputs);
    del_Array(Y_te, n_outputs);
    del_Array(Xn_tr, n_inputs);
    del_Array(Xn_te, n_inputs);
    del_Array(Yp_tr, n_outputs);
    del_Array(Yp_te, n_outputs);
    del_Array(agents);
    if (classification) {
        del_Array(table);
        del_Array(Yc_tr);
    }
    else {
        del_Array(Ys_tr, n_outputs);
        del_Array(Ys_te, n_outputs);
    }

    printf("\n\n\n===== End =====\n\n");

    return 0;
}


/* Interface between the SA solver and the ANFIS */
double interface(double *theta, int idx, Arguments args) 
{
    double J;

    // Classification problem
    if (args.Yc != nullptr) {
        J = args.agents[idx].create_model(theta, args.X, args.Yc, args.n_samples);
    }

    // Regression problem
    else {
        J = args.agents[idx].create_model(theta, args.X, args.Y, args.n_samples);
    }

    return J;
}


/* Read (comma separated) data from a file */
double **read_data(string data_file, int& nr, int& nc, bool flip)
{

    // Open file
    ifstream idf;
    idf.open(data_file, ios::in);

    // If no errors
    if (idf.is_open()) {

        string line;
        size_t pos;
        double **data;

        // Read the first line and determine the number of columns
        int cols = 0;
        getline(idf, line);
        while ((pos = line.find(',')) != string::npos) {
            line.erase(0, pos+1);
            cols++; 
        }
        cols++;         // There should be (cols-1) commas

        // Count the remaining lines
        int rows = 1;   // Previous (first) line
        while (getline(idf, line)) {
            rows++;

            //  Count the number of columns
            int count = 0;
            while ((pos = line.find(',')) != string::npos) {
                count++; 
                line.erase(0, pos+1);
            }

            // Check consistency in the number of columns
            if ((count+1) != cols) {
                printf("\nRow %d in %s", rows, data_file.c_str());
                printf("\n--> Wrong number of columns.\n\n");
                exit(EXIT_FAILURE);
            }
        }

        // Reset to the beginning of the file
        idf.clear();
        idf.seekg(0, ios::beg);

        // Read the data as they are
        if (!flip) {
            data = new_Array<double>(rows, cols);
            for (int i=0; i<rows; i++) {
                getline(idf, line);
                for (int j=0; j<cols-1; j++) {
                    pos = line.find(',');
                    data[i][j] = stod(line.substr(0, pos));
                    line.erase(0, pos+1);
                }
                data[i][cols-1] = stod(line);
            }
            nr = rows;
            nc = cols;
        }

        // Read the data and flip the resulting matrix
        else {
            data = new_Array<double>(cols, rows);
            for (int i=0; i<rows; i++) {
                getline(idf, line);
                for (int j=0; j<cols-1; j++) {
                    pos = line.find(',');
                    data[j][i] = stod(line.substr(0, pos));
                    line.erase(0, pos+1);
                }
                data[cols-1][i] = stod(line);
            }
            nr = cols;
            nc = rows;

        }

        // Close file
        idf.close();

        return data;
    }

    // Error opening the file
    else {
        printf("\n%s", data_file.c_str());
        printf("\n--> Unable to open the file.\n\n");
        exit(EXIT_FAILURE);
    }
}


/*
Builds the boundaries for the SA solver using a few simple rules.

Premise parameters:
- Means (mu) are equidistributed (starting from the min. value) along the
  input dataset and are allowed to move by <mu_delta> on each side. The value
  of <mu_delta> is expressed as a fraction of the range.
- Standard deviations (s) are initially the same for all MFs, and are given
  using a middle value <s_par[0]> and its left/right variation <s_par[1]>.
  The middle value is scaled based on the actual range of inputs.
- Exponents (c) are initially the same for all MFs, and are given using a
  range, i.e. a min. value <c_par[0]> and a max. value <c_par[1]>.

Consequent parameters:
- Coefficients (A) are given using a range, i.e. a min. value <A_par[0]>
  and a max. value <A_par[1]>.
*/
void bounds(double** X, int n_samples, int* MFs, int n_inputs, int n_outputs,
            double mu_delta, double s_par[2], double c_par[2], double A_par[2],
            double*& LB, double*& UB) 
{
    // Anfis parameters
    int n_pf = sum(MFs, n_inputs);
    int n_cf = prod(MFs, n_inputs);
    int n_var = 3 * n_pf + (n_inputs + 1) * n_cf * n_outputs;
    LB = new_Array<double>(n_var);
    UB = new_Array<double>(n_var);

    // Premise parameters (mu, s, c)
    int idx = 0;
    for (int j=0; j<n_inputs; j++) {

        // Feature (input) min, max, and range
        double X_min = value_min(X[j], n_samples);
        double X_max = value_max(X[j], n_samples);
        double X_delta = X_max - X_min;

        // Init parameters for mean and standard deviation 
        double X_step, X_start, s;
        if (MFs[j] == 1) {
            X_step = 0.0;
            X_start = (X_min + X_max) / 2.0;
            s = s_par[0];
        }
        else {
            X_step = X_delta / static_cast<double>(MFs[j] - 1);
            X_start = X_min;
            s = s_par[0] * X_step;
        }

        // Assign values to the lower (LB) and higher (UB) boundary arrays
        for (int k=0; k<MFs[j]; k++) {
            double mu = X_start + X_step * static_cast<double>(k);
            LB[idx] = mu - mu_delta * X_delta;          // mu lower limit
            UB[idx] = mu + mu_delta * X_delta;          // mu upper limit
            LB[n_pf+idx] = s - s_par[1];                // s lower limit
            UB[n_pf+idx] = s + s_par[1];                // s upper limit
            LB[2*n_pf+idx] = c_par[0];                  // c lower limit
            UB[2*n_pf+idx] = c_par[1];                  // c upper limit
            idx++;
        }
        for (int k=3*n_pf; k<n_var; k++) {
            LB[k] = A_par[0];                           // A lower limit
            UB[k] = A_par[1];                           // A upper limit
        }        
    }

    return;
}
