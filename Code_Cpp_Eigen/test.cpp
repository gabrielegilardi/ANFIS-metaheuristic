/*
Multivariate Regression and Classification Using an Adaptive Neuro-Fuzzy
Inference System (Takagi-Sugeno) and Simulated Annealing Optimization.

Copyright (c) 2021 Gabriele Gilardi


Features
--------
- The code has been written in C++ using the Eigen library (ver. 3.4.0) and
  tested using g++ 11.2.0 in MinGW-W64 9.0.0-r1.
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
- File "utils.cpp" consists of several helper functions.
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

- Eigen template library for linear algebra
  @ https://eigen.tuxfamily.org
*/

/* Headers */
#include <fstream>
#include "utils.hpp"
#include "ANFIS.hpp"

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
    ArrayXi IntVar;
};

/* Structure to pass the data to the ANFIS */
struct Arguments {
    ArrayXXd X;                 // Training dataset inputs/features
    ArrayXXd Y;                 // Training dataset labels/outputs (regression)
    ArrayXi Yc;                 // Training dataset classes (classification)
    AnfisType* agents;          // Array of ANFIS agents
};

/* Prototypes (local functions) */
ArrayXd sa(ArrayXd (*func)(ArrayXXd, Arguments), ArrayXd LB, ArrayXd UB,
           Parameters p, Arguments args, mt19937_64& gen);
ArrayXd interface(ArrayXXd theta, Arguments args) ;
ArrayXXd read_data(string data_file, int& rows, int& cols, bool flip);
void bounds(ArrayXXd X, ArrayXi MFs, int n_out, double mu_delta, double s_par[2],
            double c_par[2], double A_par[2], ArrayXd& LB, ArrayXd& UB);


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
    ArrayXi MFs;
    Parameters p;
    mt19937_64 gen(seed);

    // Single-label continuous problem example
    if (example == "plant") {
        // Dataset: 4 features (inputs), 1 label (output), 9568 samples
        // ANFIS: layout of [1, 1, 1, 1], 17 variables
        // Predicted/actual correlation values: 0.964 (training), 0.962 (test)
        // https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
        data_file = "plant_dataset.csv";
        MFs.setOnes(4);
    }

    // Multi-label continuous problem example
    else if (example == "stock") {
        // Dataset: 3 features (inputs), 2 labels (outputs), 536 samples
        // ANFIS: layout of [2, 2, 2], 82 variables
        // Predicted/actual correlation values: 0.918 (training), 0.912 (test)
        // https://archive.ics.uci.edu/ml/datasets/ISTANBUL+STOCK+EXCHANGE
        data_file = "stock_dataset.csv";
        MFs.setConstant(3, 2);
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
        // Predicted/actual accuracy values: 58.2% (training), 56.7% (test).
        // https://archive.ics.uci.edu/ml/datasets/Wine+Quality
        data_file = "wine_dataset.csv";
        // MFs.resize(2);
        MFs.setZero(2);
        MFs << 3, 2;
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
        // Predicted/actual accuracy values: 97.6% (training), 97.4% (test).
        // https://archive.ics.uci.edu/ml/datasets/HTRU2
        data_file = "pulsar_dataset.csv";
        MFs.setZero(3);
        MFs << 3, 4, 2;
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

    // Read (comma-separated) data from a file
    int n_rows, n_cols;
    ArrayXXd data = read_data(data_file, n_rows, n_cols, false);
    int n_samples = n_rows;

    // Randomly shuffle the indexes and build the shuffled data matrix
    ArrayXi idx_shuffle = shuffle(n_samples, gen);
    ArrayXXd data_shuffle = data(idx_shuffle, all);

    // For a classification problem build the class table (classes must always
    // be on the last column)
    int n_classes, n_outputs, n_inputs;
    n_inputs = MFs.size();
    ArrayXd table;
    if (classification) {
        table = build_class_table(data.col(n_cols-1), tol);
        n_classes = table.size();
        n_outputs = 1;
    }

    // In a regression problem the outputs are always in the last columns
    else {
        n_classes = 0;
        n_outputs = n_cols - n_inputs;
    }

    // Build the splitted datasets ( training dataset: 0 --> samples_tr-1,
    // test dataset: samples_tr --> n_samples-1)
    int samples_tr = int(nearbyint(split_factor * double(n_samples)));
    int samples_te = n_samples - samples_tr;
    ArrayXXd X_tr = data_shuffle.topLeftCorner(samples_tr, n_inputs);
    ArrayXXd Y_tr = data_shuffle.topRightCorner(samples_tr, n_outputs);
    ArrayXXd X_te = data_shuffle.bottomLeftCorner(samples_te, n_inputs);
    ArrayXXd Y_te = data_shuffle.bottomRightCorner(samples_te, n_outputs);

    // Normalize inputs (cols) using the training dataset mean and std
    ArrayXXd Xn_tr, Xn_te;
    Xn_tr.setZero(samples_tr, n_inputs);
    Xn_te.setZero(samples_te, n_inputs);
    for (int j=0; j<n_inputs; j++) {
        double mu = X_tr.col(j).mean();
        double sigma = stdev(X_tr.col(j));
        Xn_tr.col(j) = normalize(X_tr.col(j), mu, sigma);
        Xn_te.col(j) = normalize(X_te.col(j), mu, sigma);
    }

    // For classification get the classes in the training dataset output (the
    // original classes are re-numbered 0 --> n_classes-1)
    int n_out;
    ArrayXi Yc_tr;
    ArrayXXd Ys_tr, Ys_te;
    if (classification) {
        Yc_tr = get_classes(Y_tr.col(0), table, tol);
        n_out = n_classes;          // Actual number of outputs used by ANFIS
    }

    // For regression scale the outputs (rows) to the interval [-1, 1], to
    // reduce the range of <A> 
    else {
        Ys_tr.setZero(samples_tr, n_outputs);
        Ys_te.setZero(samples_te, n_outputs);
        for (int j=0; j<n_outputs; j++) {
            double Ymin = Y_tr.col(j).minCoeff();
            double Ymax = Y_tr.col(j).maxCoeff();
            Ys_tr.col(j) = scale(Y_tr.col(j), Ymin, Ymax);
            Ys_te.col(j) = scale(Y_te.col(j), Ymin, Ymax);
        }
        n_out = n_outputs;
    }

    // Init ANFIS agents
    AnfisType* agents = new AnfisType [p.nPop];
    for (int i=0; i<p.nPop; i++) {
        agents[i].init(MFs, n_out);
    }

    // Init arguments to be passed
    Arguments args;
    args.X = Xn_tr;
    if (classification) {
        args.Yc = Yc_tr;
    }
    else {
        args.Y = Ys_tr;
    }
    args.agents = agents;

    // Init the SA search space boundaries (arrays LB and UB)
    ArrayXd LB, UB;
    bounds(Xn_tr, MFs, n_out, mu_delta, s_par, c_par, A_par, LB, UB);

    // Show dataset info
    printf("\n\n===== Dataset info =====");
    printf("\n- Example: %s", example.c_str());
    printf("\n- File: %s", data_file.c_str());
    printf("\n- Total samples: %d", n_samples);
    if (classification) {
        printf("\n- Number of classes: %d", n_classes);
        printf("\n- Class table:");
        for (int i=0; i<n_classes; i++) {
            printf(" %g ", table(i));
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
    ArrayXd (*func)(ArrayXXd, Arguments);
    func = interface;
    ArrayXd best_sol = sa(func, LB, UB, p, args, gen);
    delete[] agents;

    // Re-build the ANFIS with the best solution and evaluate datasets
    double J;
    ArrayXXd Yp_tr, Yp_te;
    AnfisType best_agent;
    best_agent.init(MFs, n_out);
    if (classification) {
        J = best_agent.create_model(best_sol, Xn_tr, Yc_tr);
        Yp_tr = best_agent.eval_data(Xn_tr, table);
        Yp_te = best_agent.eval_data(Xn_te, table);
    }
    else {
        J = best_agent.create_model(best_sol, Xn_tr, Ys_tr);
        Yp_tr = best_agent.eval_data(Xn_tr);
        Yp_te = best_agent.eval_data(Xn_te);
    }
    // Show results
    printf("\n\n===== Results =====");
    printf("\nJ = %g", J);
    printf("\n\n    mu        c         s");
    for (int i=0; i<n_pf; i++) {
        printf("\n%8.4f  %8.4f  %8.4f", best_agent.mu(i), best_agent.s(i),
                                        best_agent.c(i));
    }
    printf("\n\nA");
    for (int i=0; i<n_inputs+1; i++) {
        printf("\n");
        for (int j=0; j<n_cf*n_out; j++) {
            printf("%10.4f ", best_agent.A(i, j));
        }
    }
    printf("\n\n\n===== Stats =====");
    if (classification) {
        printf("\nAcc. (tr) = %g", accuracy(Yp_tr.reshaped(), Y_tr.reshaped()));
        printf("\nAcc. (te) = %g", accuracy(Yp_te.reshaped(), Y_te.reshaped()));
        printf("\nCorr. (tr) = %g", calc_corr(Yp_tr.reshaped(), Y_tr.reshaped()));
        printf("\nCorr. (te) = %g", calc_corr(Yp_te.reshaped(), Y_te.reshaped()));
    }
    else {
        printf("\nCorr. (tr) = %g", calc_corr(Yp_tr.reshaped(), Ys_tr.reshaped()));
        printf("\nCorr. (te) = %g", calc_corr(Yp_te.reshaped(), Ys_te.reshaped()));
        printf("\nRMSE (tr) = %g", rmse(Yp_tr.reshaped(), Ys_tr.reshaped()));
        printf("\nRMSE (te) = %g", rmse(Yp_te.reshaped(), Ys_te.reshaped()));
    }

    printf("\n\n\n===== End =====\n\n");

    return 0;
}


/* Interface between the SA solver and the ANFIS */
ArrayXd interface(ArrayXXd theta, Arguments args) 
{
    int n_agents = theta.rows();

    ArrayXd J;
    J.setZero(n_agents);

    // Classification problem
    if (args.Yc.size() != 0) {
        for (int i=0; i<n_agents; i++) {
            J(i) = args.agents[i].create_model(theta.row(i), args.X, args.Yc);
        }
    }

    // Regression problem
    else {
        for (int i=0; i<n_agents; i++) {
            J(i) = args.agents[i].create_model(theta.row(i), args.X, args.Y);
        }
    }

    return J;
}


/* Read (comma separated) data from a file */
ArrayXXd read_data(string data_file, int& nr, int& nc, bool flip)
{

    // Open file
    ifstream idf;
    idf.open(data_file, ios::in);

    // If no errors
    if (idf.is_open()) {

        string line;
        size_t pos;
        ArrayXXd data;

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
            data.setZero(rows, cols);
            for (int i=0; i<rows; i++) {
                getline(idf, line);
                for (int j=0; j<cols-1; j++) {
                    pos = line.find(',');
                    data(i, j) = stod(line.substr(0, pos));
                    line.erase(0, pos+1);
                }
                data(i, cols-1) = stod(line);
            }
            nr = rows;
            nc = cols;
        }

        // Read the data and flip the resulting matrix
        else {
            data.setZero(cols, rows);
            for (int i=0; i<rows; i++) {
                getline(idf, line);
                for (int j=0; j<cols-1; j++) {
                    pos = line.find(',');
                    data(j, i) = stod(line.substr(0, pos));
                    line.erase(0, pos+1);
                }
                data(cols-1, i) = stod(line);
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
void bounds(ArrayXXd X, ArrayXi MFs, int n_out, double mu_delta, double s_par[2],
            double c_par[2], double A_par[2], ArrayXd& LB, ArrayXd& UB) 
{
    int n_inputs = MFs.size();
    int n_samples = X.rows();

    // Anfis parameters
    int n_pf = MFs.sum();
    int n_cf = MFs.prod();
    int n_var = 3 * n_pf + (n_inputs + 1) * n_cf * n_out;
    
    LB.setZero(n_var);
    UB.setZero(n_var);

    // Premise parameters (mu, s, c)
    int idx = 0;
    for (int j=0; j<n_inputs; j++) {

        // Feature (input) min, max, and range
        double Xmin = X.col(j).minCoeff();
        double Xmax = X.col(j).maxCoeff();
        double Xdelta = Xmax - Xmin;

        // Init parameters for mean and standard deviation 
        double Xstep, Xstart, s;
        if (MFs(j) == 1) {
            Xstep = 0.0;
            Xstart = (Xmin + Xmax) / 2.0;
            s = s_par[0];
        }
        else {
            Xstep = Xdelta / static_cast<double>(MFs(j) - 1);
            Xstart = Xmin;
            s = s_par[0] * Xstep;
        }

        // Assign values to the lower (LB) and higher (UB) boundary arrays
        for (int k=0; k<MFs(j); k++) {
            double mu = Xstart + Xstep * static_cast<double>(k);
            LB(idx) = mu - mu_delta * Xdelta;          // mu lower limit
            UB(idx) = mu + mu_delta * Xdelta;          // mu upper limit
            LB(n_pf+idx) = s - s_par[1];                // s lower limit
            UB(n_pf+idx) = s + s_par[1];                // s upper limit
            LB(2*n_pf+idx) = c_par[0];                  // c lower limit
            UB(2*n_pf+idx) = c_par[1];                  // c upper limit
            idx++;
        }
        for (int k=3*n_pf; k<n_var; k++) {
            LB(k) = A_par[0];                           // A lower limit
            UB(k) = A_par[1];                           // A upper limit
        }        
    }

    return;
}
