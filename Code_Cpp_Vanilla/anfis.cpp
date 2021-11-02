/*
Multivariate Regression and Classification Using an Adaptive Neuro-Fuzzy
Inference System (Takagi-Sugeno) and Simulated Annealing Optimization.

Copyright (c) 2021 Gabriele Gilardi


X           (n_inputs, n_samples)           Input dataset (training)
Y           (n_outputs, n_samples)          Output dataset (training)
Xp          (n_inputs, n_samples)           Input dataset (prediction)
Yp          (n_labels, n_samples)           Output dataset (prediction)
theta       (n_var, 1)                      Unrolled parameters
mu          (n_pf, 1)                       Mean (premise MFs)
s           (n_pf, 1)                       Standard deviation (premise MFs)
c           (n_pf, 1)                       Exponent (premise MFs)
A           (n_inputs+1, n_cf*n_outputs)    Coefficients (consequent MFs)
pf          (n_pf, 1)                       Premise MFs value
W           (n_cf, 1)                       Firing strenght value
Wr          (n_cf, 1)                       Firing strenght ratios
cf          (n_cf, 1)                       Consequent MFs value
f           (n_outputs, n_samples)          ANFIS output
combs       (n_inputs, n_cf)                Combinations of premise MFs

n_samples           Number of samples
n_inputs            Number of features/inputs
n_outputs           Number of labels/classes/outputs
n_var               Number of variables
n_mf                Number of premise MFs of each feature
n_pf                Total number of premise MFs = sum(MFs)
n_cf                Total number of consequent MFs = prod(MFs)

Notes:
- MF stands for membership function.
- premise (membership) functions are generalize Bell function defined by mean
  <mu>, standard deviation <s>, and exponent <c>.
- consequent (membership) functions are hyperplanes defined by <n_inputs+1>
  coefficients each.
- In a classification problem the number of outputs is the same as the number
  of classes.
*/

/* Headers and namespaces */
#include <cmath>
#include <cstdio>

#include "ANFIS.hpp"
#include "templates.hpp"              // Namespace "std" is in this file


/* ===== Helper functions ===== */

/* Returns an array (table) will all the possible classes */
double* build_class_table(double* Y, int nel, int& n_classes, double tol)
{
    bool* flag = new_Array<bool>(nel);
    set_Array(flag, nel, false);
    flag[0] = true;                     // 1st element is for sure unique

    // Identify the unique classes grouping the quantities base on <tol>
    n_classes = 1;
    for (int i=1; i<nel; i++) {
        bool duplicate = false;

        // Check if the i-th element is unique 
        for (int j=0; j<i; j++) {
            if (flag[j] && (fabs(Y[j] - Y[i]) < tol)) {
                duplicate = true;       // Not unique 
                break;
            }
        }

        // If the element is unique count it and mark it
        if (!duplicate) {
            n_classes++;
            flag[i] = true;
        }
    }

    // Build the array (table) with all the unique elements
    int count = 0;
    double* table = new_Array<double>(n_classes);
    for (int i=0; i<nel; i++) {
        if (flag[i]) {
            table[count] = Y[i];
            count++;
        }
    }

    del_Array(flag);

    return table;
}

/*
Converts the output array of a classification problem to a number in the
interval [0, n_classes-1]
*/
int* get_classes(double* Y, int nel, double* table, int n_classes, double tol)
{
    int* Yc = new_Array<int>(nel);

    for (int i=0; i<nel; i++) {
        bool found = false;

        // Search the i-th element in the class table
        for (int j=0; j<n_classes; j++) {
            if (fabs(table[j] - Y[i]) < tol) {
                Yc[i] = j;
                found = true;
                break;
            }
        }

        // Every element must be in the class table if the same <tol> is used
        if (!found) {
            printf("\nClass %g", Y[i]);
            printf("\n--> Not found in the class table.\n\n");
            exit(EXIT_FAILURE);
        }
    }

    return Yc;
}


/* ===== ANFIS class ===== */

/* Constructor */
AnfisType::AnfisType() {

    // DO nothing, everything is initialized in "init"

}

/* Initializes class. */
void AnfisType::init(int n_inputs0, int n_outputs0, int *MFs0)
{
    // Passed arguments
    n_inputs = n_inputs0;
    n_outputs = n_outputs0;
    MFs = new_Array<int>(n_inputs);
    copy_Array(MFs, n_inputs, MFs0);

    // Derived quantities
    n_pf = sum(MFs, n_inputs);
    n_cf = prod(MFs, n_inputs);
    n_var = 3 * n_pf + (n_inputs + 1) * n_cf * n_outputs;

    // Build all the possible combination of premise functions
    build_combinations();
    
    // Premise function parameters
    mu = new_Array<double>(n_pf); 
    s = new_Array<double>(n_pf); 
    c = new_Array<double>(n_pf);

    // Consequent function parameters
    A = new_Array<double>(n_inputs+1, n_cf*n_outputs);

    return;
}

/* Creates the model for a regression (continuos) problem */
double AnfisType::create_model(double* theta, double** X, double** Y, int n_samples)
{
    // Build parameters mu, s, c, and A
    build_param(theta);

    // Calculate output
    double** f = forward_steps(X, n_samples);    

    // Calculate cost function
    double err = 0.0;
    for (int j=0; j<n_outputs; j++) {
        for (int i=0; i<n_samples; i++) {
            double d = f[j][i] - Y[j][i];
            err += d * d;
        }
    }

    del_Array(f, n_outputs);

    return (err / 2.0);
}

/* Creates the model for a classification problem */
double AnfisType::create_model(double* theta, double** X, int* Yc, int n_samples)
{
    // Build the matrix <Y> actually used as output. Array <Yc> has dimensions
    // (n_samples, ) while <Y> has dimension (n_classes, n_samples). In this 
    // matrix, Y[j,i] = 1 specifies that the i-th sample belongs to the j-th
    // class.
    double** Y = new_Array<double>(n_outputs, n_samples);
    set_Array(Y, n_outputs, n_samples, 0.0);        // n_outputs = n_classes
    for (int i=0; i<n_samples; i++) {
        Y[Yc[i]][i] = 1.0;
    }

    // Build parameters mu, s, c, and A
    build_param(theta);

    // Calculate output
    double** f = forward_steps(X, n_samples);

    // Calculate cost function (the activation value is determined in the
    // logsig function)
    double err = 0.0;
    for (int j=0; j<n_outputs; j++) {
        for (int i=0; i<n_samples; i++) {
            err += (1.0 - Y[j][i]) * f[j][i] - logsig(f[j][i]);
        }
    }

    del_Array(f, n_outputs);
    del_Array(Y, n_outputs);

    return (err / double(n_samples));
}

/* Evaluates a model for a regression (continuos) problem */
double** AnfisType::eval_data(double** Xp, int n_samples)
{
    // Calculate output
    double** Yp = forward_steps(Xp, n_samples);

    return Yp;
}

/* Evaluates a model for a classification problem */
double** AnfisType::eval_data(double** Xp, int n_samples, double* table)
{
    // Calculate output
    double** f = forward_steps(Xp, n_samples);

    // Loop over each sample
    double* fa = new_Array<double>(n_outputs);      // n_outputs = n_classes
    double** Yp = new_Array<double>(1, n_samples);
    for (int i=0; i<n_samples; i++) {

        // Activation values
        for (int j=0; j<n_outputs; j++) {
            fa[j] = f_activation(f[j][i]);
        }

        // Class with max. probability expressed as index in [0, n_classes-1]
        int idx = idx_max(fa, n_outputs);

        // Assign best result and return the original class (for consistency
        // Yp is created and reterned as a row-vector)
        Yp[0][i] = table[idx];
    }

    del_Array(fa);

    return Yp;
}

/* Show ANFIS info */
void AnfisType::info()
{
    printf("\n ");
    printf("\n\n===== ANFIS info =====");
    printf("\n- Inputs: %d", n_inputs);
    printf("\n- Outputs: %d", n_outputs);
    printf("\n- Variables: %d", n_var);
    printf("\n- MF Layout:");
    for (int i=0; i<n_inputs; i++) {
        printf(" %d ", MFs[i]);
    }
    printf("\n- Number of PF: %d", n_pf);
    printf("\n- Number of CF: %d", n_cf);
    printf("\n ");

    return;
}

/* Class destructor */
AnfisType::~AnfisType()
{
    del_Array(MFs);
    del_Array(combs, n_inputs);
    del_Array(mu);
    del_Array(s);
    del_Array(c);
    del_Array(A, n_inputs+1);
}

/* 
Build all the possible combination of premise functions

For example if <n_mf> = [3, 2], the MF indexes for the first feature would be
[0, 1, 2] and for the second feature would be [3, 4]. The resulting combinations
would be <combs> = [[0 0 1 1 2 2],
                    [3 4 3 4 3 4]].

*/
void AnfisType::build_combinations()
{
    int* ics = cumsum(MFs, n_inputs);
    int* icp = cumprod(MFs, n_inputs);
    combs = new_Array<int>(n_inputs, n_cf);

    // Build the first row
    int idx = 0;
    int steps = n_cf / icp[0];
    for (int k=0; k<ics[0]; k++) {
        for (int j=0; j<steps; j++) {
            combs[0][idx] = k;
            idx++;
        }
    }

    // Recursively build the other rows (if any)
    for (int m=1; m<n_inputs; m++) {
        idx = 0;
        steps = n_cf / icp[m];
        for (int i=0; i<icp[m-1]; i++) {
            for (int k=ics[m-1]; k<ics[m]; k++) {
                for (int j=0; j<steps; j++) {
                    combs[m][idx] = k;
                    idx++;
                }
            }
        }
    }

    del_Array(ics);
    del_Array(icp);

    return;
}

/* Builds the premise/consequent parameters mu, s, c, and A */
void AnfisType::build_param(double* theta)
{
    // Premise parameters
    for (int i=0; i<n_pf; i++) {
        mu[i] = theta[i];
        s[i] = theta[i+n_pf];
        c[i] = theta[i+2*n_pf];
    }

    // Consequent parameters
    int idx = 3 * n_pf;
    for (int i=0; i<n_inputs+1; i++) {
        for (int j=0; j<n_cf*n_outputs; j++) {
            A[i][j] = theta[idx];
            idx++;
        }
    }

    return;
}

/*
Calculate the output of the ANFIS layers giving the premise and consequent
parameters and the input dataset.
*/
double** AnfisType::forward_steps(double** X, int n_samples)
{
    // Allocate dynamic arrays
    double* pf = new_Array<double>(n_pf);
    double* W = new_Array<double>(n_cf);
    double* Wr = new_Array<double>(n_cf);
    double* cf = new_Array<double>(n_cf);
    double** f = new_Array<double>(n_outputs, n_samples);

    // Cumulative sum (and corresponding dynamic array allocation)
    int* ics = cumsum(MFs, n_inputs);

    // Loop over each samples
    for (int i=0; i<n_samples; i++) {

        // Layer 1: premise functions (pf)
        int idx = 0;
        for (int j=0; j<n_pf; j++) {
            if (j >= ics[idx]) {
                idx++;
            }
            double tmp = (X[idx][i] - mu[j]) / s[j];
            pf[j] = 1.0 / (1.0 + pow(tmp * tmp, c[j]));
        }

        // Layer 2: firing strenght (W)
        for (int j=0; j<n_cf; j++) {
            double tmp = 1.0;
            for (int k=0; k<n_inputs; k++) {
                tmp *= pf[combs[k][j]];
            }
            W[j] = tmp;
        }

        // Layer 3: firing strenght ratios (Wr)
        double tmp = sum(W, n_cf);
        for (int j=0; j<n_cf; j++) {
            Wr[j] = W[j] / tmp;
        }

        // Layer 4 and 5: consequent functions (cf) and output (f)
        for (int m=0; m<n_outputs; m++) {
            int idx = m * n_cf;
    
            // cf
            for (int j=0; j<n_cf; j++) {
                double tmp = A[0][j+idx];
                for (int k=1; k<=n_inputs; k++) {
                    tmp += X[k-1][i] * A[k][j+idx];
                }
                cf[j] = Wr[j] * tmp;
            }

            // f
            f[m][i] = sum(cf, n_cf);
        }
    }

    del_Array(pf);
    del_Array(W);
    del_Array(Wr);
    del_Array(cf);
    del_Array(ics);

    return f;
}

/*
Numerically stable version of the sigmoid function.

Ref.: http://fa.bianp.net/blog/2019/evaluate_logistic/#sec3
*/
double AnfisType::f_activation(double z)
{
    double a;

    // Value in [0, +inf)
    if (z >= 0.0) {
        a = 1.0 / (1.0 + exp(-z));
    }

    // Value in (-inf, 0)
    else {
        a = exp(z) / (1.0 + exp(z));
    }

    return a;
}

/*
Numerically stable version of the log-sigmoid function.

Ref.: http://fa.bianp.net/blog/2019/evaluate_logistic/#sec3
*/
double AnfisType::logsig(double z)
{
    double a;

    // Value in (-inf, -33.3)
    if (z < -33.3) {
        a = z;
    }

    // Value in [-33.3, -18.0)
    else if ((z >= -33.3) && (z < -18.0)) {
        a = z - exp(z);
    }

    // Value in [-18.0, +37.0)
    else if ((z >= -18.0) & (z < 37.0)) {
        a = - log(1.0 + exp(-z));
    }

    // Value in [+37.0, +inf)
    else {
        a = - exp(-z);
    }

    return a;
}
